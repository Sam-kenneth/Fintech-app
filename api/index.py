import os
import json
import requests
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import logging
from datetime import datetime
import zipfile
from io import BytesIO

# ✅ IMPORT LLM PROVIDER (NEW)
from .llm_provider import get_llm_provider
from .financial_api_integration import FinancialDataIntegration, api_data_to_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates')

# Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'fintech_uploads')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'zip'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB for multiple files
MAX_FILES = 50  # Maximum files per customer

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for Vercel frontend
try:
    from flask_cors import CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
except ImportError:
    logger.warning("flask-cors not installed, CORS disabled")

# Initialize embeddings model
try:
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Embeddings model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embeddings model: {e}")
    embed_model = None

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# Store customer sessions
customer_sessions = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class CustomerSession:
    """Manages a customer's documents and analysis"""
    def __init__(self, customer_id):
        self.customer_id = customer_id
        self.files = {}  # filename -> file_path (uploaded files)
        self.ocr_texts = {}  # filename -> extracted text
        self.api_data = {}  # api_type -> api_response (API fetched data)
        self.vector_db = None
        self.all_text = ""
        self.created_at = datetime.now()
        self.analysis_results = {}
        self.data_sources = {
            'uploaded_files': [],
            'api_sources': []
        }
    
    def add_file(self, filename, file_path):
        """Add a file to the session"""
        self.files[filename] = file_path
        if filename not in self.data_sources['uploaded_files']:
            self.data_sources['uploaded_files'].append(filename)
    
    def add_ocr_text(self, filename, text):
        """Add OCR extracted text"""
        self.ocr_texts[filename] = text
        self.all_text += f"\n\n{'='*80}\nDOCUMENT: {filename}\n{'='*80}\n{text}"
    
    def add_api_data(self, api_type, api_data):
        """Add API fetched data to session"""
        self.api_data[api_type] = api_data
        
        # Convert API data to text format
        text_content = self._api_data_to_text(api_data, api_type)
        
        # Add to OCR texts and all_text
        filename = f"{api_type}_api_fetch"
        self.ocr_texts[filename] = text_content
        self.all_text += f"\n\n{'='*80}\nDATA SOURCE: {api_type.upper()} (API FETCHED)\n{'='*80}\n{text_content}"
        
        # Track data source
        if api_type not in self.data_sources['api_sources']:
            self.data_sources['api_sources'].append(api_type)
        
        logger.info(f"Added {api_type} API data to session {self.customer_id}")
    
    def _api_data_to_text(self, data, doc_type):
        """Convert API response data to text for document processing"""
        text = f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        def dict_to_text(d, indent=0):
            result = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    result += f"{'  '*indent}{key}:\n"
                    result += dict_to_text(value, indent+1)
                elif isinstance(value, list):
                    result += f"{'  '*indent}{key}:\n"
                    for item in value:
                        if isinstance(item, dict):
                            result += dict_to_text(item, indent+1)
                        else:
                            result += f"{'  '*(indent+1)}- {item}\n"
                else:
                    result += f"{'  '*indent}{key}: {value}\n"
            return result
        
        text += dict_to_text(data)
        return text
    
    def create_vector_db(self):
        """Create FAISS vector database from all texts (uploaded + API)"""
        if not self.all_text.strip():
            return False
        try:
            self.vector_db = FAISS.from_texts([self.all_text], embed_model)
            logger.info(f"Vector DB created for customer {self.customer_id}")
            logger.info(f"Data sources - Uploaded: {len(self.data_sources['uploaded_files'])}, "
                       f"API: {len(self.data_sources['api_sources'])}")
            return True
        except Exception as e:
            logger.error(f"Failed to create vector DB: {e}")
            return False
    
    def get_session_info(self):
        """Get session information"""
        return {
            'customer_id': self.customer_id,
            'uploaded_file_count': len(self.data_sources['uploaded_files']),
            'api_source_count': len(self.data_sources['api_sources']),
            'uploaded_files': self.data_sources['uploaded_files'],
            'api_sources': self.data_sources['api_sources'],
            'total_files': len(self.files) + len(self.api_data),
            'created_at': self.created_at.isoformat(),
            'total_text_length': len(self.all_text),
            'has_vector_db': self.vector_db is not None
        }


def perform_ocr(file_path):
    """Extract text from document using OCR.space API"""
    try:
        logger.info(f"Performing OCR on {file_path}")
        
        with open(file_path, 'rb') as f:
            r = requests.post(
                'https://api.ocr.space/parse/image',
                files={'filename': f},
                data={
                    'apikey': 'K87899142791569',
                    'language': 'eng'
                },
                timeout=30
            )
        
        result = r.json()
        
        if result.get('IsErroredOnProcessing'):
            logger.error(f"OCR error: {result.get('ErrorMessage')}")
            return None
        
        text = result.get('ParsedText', '')
        logger.info(f"OCR extracted {len(text)} characters")
        return text
    
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return None


# ✅ UPDATED FUNCTION 1: Using LLM Provider
def analyze_multi_document_risk(context, customer_id, file_list):
    """Generate comprehensive risk assessment for multiple documents"""
    try:
        llm = get_llm_provider()  # ✅ GET LLM PROVIDER
        
        files_str = "\n".join([f"- {f}" for f in file_list])
        
        comprehensive_prompt = f"""
You are a fintech credit risk analyst reviewing a customer's complete financial profile.

CUSTOMER ID: {customer_id}
DOCUMENTS ANALYZED:
{files_str}

EXTRACTED CONTEXT FROM ALL DOCUMENTS:
{context[:3000]}

TASK: Provide a comprehensive risk assessment based on ALL available documents.

Structure your response as JSON with the following keys:
1. "customer_profile" - Brief profile based on documents
2. "risk_score" - Overall risk score (0-100, where 100 is highest risk)
3. "risk_category" - Low / Moderate / High / Critical
4. "income_analysis" - Income stability and consistency
5. "payment_history" - EMI and payment patterns
6. "debt_to_income_ratio" - Assessment if available
7. "red_flags" - List of concerning patterns
8. "positive_factors" - List of positive indicators
9. "missing_documents" - Any important docs that might be needed
10. "recommendations" - Lending recommendation
11. "confidence_level" - Confidence in assessment (0-100%)
12. "detailed_reasoning" - Step-by-step analysis

Be thorough and consider all documents together. Provide specific data points from the documents.
"""
        
        response = llm.generate_analysis(comprehensive_prompt)  # ✅ USE LLM PROVIDER
        return response  # ✅ RETURN DIRECTLY (not response['response'])
    except Exception as e:
        logger.error(f"Multi-document analysis error: {e}")
        return {"error": str(e)}


# ✅ UPDATED FUNCTION 2: Using LLM Provider
def generate_customer_report(session):
    """Generate a comprehensive customer report"""
    try:
        llm = get_llm_provider()  # ✅ GET LLM PROVIDER
        
        report_prompt = f"""
Generate a professional credit assessment report for a loan application.

CUSTOMER INFORMATION:
- ID: {session.customer_id}
- Documents: {', '.join(session.files.keys())}
- Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DOCUMENT SUMMARIES:
"""
        
        for filename, text in session.ocr_texts.items():
            summary = text[:500] + "..." if len(text) > 500 else text
            report_prompt += f"\n{filename}:\n{summary}\n"
        
        report_prompt += """

Generate a professional report with:
1. Executive Summary
2. Applicant Profile
3. Financial Analysis
4. Credit Risk Assessment
5. Key Observations
6. Final Recommendation

Format as clear, professional text suitable for credit decisions.
"""
        
        response = llm.generate_analysis(report_prompt)  # ✅ USE LLM PROVIDER
        return response  # ✅ RETURN DIRECTLY (not response['response'])
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return str(e)


# ✅ UPDATED FUNCTION 3: Using LLM Provider
def extract_financial_metrics(context):
    """Extract key financial metrics from documents"""
    try:
        llm = get_llm_provider()  # ✅ GET LLM PROVIDER
        
        metrics_prompt = f"""
Extract key financial metrics from the following documents:

{context[:2000]}

Return as JSON with:
- "monthly_income": estimated monthly income
- "total_debt": estimated total outstanding debt
- "credit_score": if mentioned
- "employment_status": employed/self-employed/etc
- "debt_obligations": list of known debt obligations
- "savings": available savings if mentioned
- "loan_amount_requested": if mentioned
- "tenure_requested": if mentioned
"""
        
        response = llm.generate_analysis(metrics_prompt)  # ✅ USE LLM PROVIDER
        return response  # ✅ RETURN DIRECTLY (not response['response'])
    except Exception as e:
        logger.error(f"Metrics extraction error: {e}")
        return str(e)


# ===== FLASK ROUTES (ALL UNCHANGED) =====

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'embeddings_available': embed_model is not None,
        'llm_provider_available': True
    }), 200


@app.route('/api/create-session', methods=['POST'])
def create_session():
    """Create a new customer session"""
    try:
        data = request.json
        customer_id = data.get('customer_id', '').strip()
        customer_name = data.get('customer_name', '').strip()
        
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400
        
        # Create new session
        session = CustomerSession(customer_id)
        customer_sessions[customer_id] = session
        
        logger.info(f"Created session for customer {customer_id}")
        
        return jsonify({
            'success': True,
            'customer_id': customer_id,
            'customer_name': customer_name,
            'message': f'Session created for {customer_name or customer_id}'
        }), 200
    
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-multiple', methods=['POST'])
def upload_multiple_files():
    """Handle multiple file uploads for a customer"""
    try:
        customer_id = request.form.get('customer_id')
        
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400
        
        if customer_id not in customer_sessions:
            return jsonify({'error': 'Customer session not found. Create session first.'}), 404
        
        session = customer_sessions[customer_id]
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        errors = []
        
        if len(files) > MAX_FILES:
            return jsonify({'error': f'Maximum {MAX_FILES} files allowed'}), 400
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                errors.append(f"{file.filename}: Invalid file type")
                continue
            
            # Handle ZIP files - extract and process
            if file.filename.endswith('.zip'):
                try:
                    zip_buffer = BytesIO(file.read())
                    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                        for zip_file in zip_ref.namelist():
                            if allowed_file(zip_file) and not zip_file.endswith('/'):
                                extracted_data = zip_ref.read(zip_file)
                                filename = secure_filename(zip_file)
                                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{customer_id}_{filename}")
                                
                                with open(file_path, 'wb') as f:
                                    f.write(extracted_data)
                                
                                session.add_file(filename, file_path)
                                uploaded_files.append(filename)
                except Exception as e:
                    errors.append(f"{file.filename}: {str(e)}")
            else:
                # Regular file upload
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{customer_id}_{filename}")
                file.save(file_path)
                
                session.add_file(filename, file_path)
                uploaded_files.append(filename)
        
        logger.info(f"Uploaded {len(uploaded_files)} files for customer {customer_id}")
        
        return jsonify({
            'success': True,
            'uploaded_files': uploaded_files,
            'total_files': len(session.files),
            'errors': errors if errors else None,
            'message': f'{len(uploaded_files)} files uploaded successfully'
        }), 200
    
    except Exception as e:
        logger.error(f"Upload Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process-all', methods=['POST'])
def process_all_documents():
    """Process all documents for a customer: OCR -> RAG -> Analysis"""
    try:
        data = request.json
        customer_id = data.get('customer_id')
        
        if not customer_id or customer_id not in customer_sessions:
            return jsonify({'error': 'Invalid customer session'}), 400
        
        session = customer_sessions[customer_id]
        
        if not session.files:
            return jsonify({'error': 'No files uploaded for this customer'}), 400
        
        logger.info(f"Processing {len(session.files)} files for customer {customer_id}")
        
        processing_results = {
            'successful': [],
            'failed': [],
            'total': len(session.files)
        }
        
        # Step 1: OCR all files
        for filename, file_path in session.files.items():
            try:
                logger.info(f"OCR processing: {filename}")
                ocr_text = perform_ocr(file_path)
                
                if ocr_text:
                    session.add_ocr_text(filename, ocr_text)
                    processing_results['successful'].append(filename)
                    logger.info(f"OCR completed: {filename}")
                else:
                    processing_results['failed'].append({
                        'file': filename,
                        'reason': 'OCR extraction failed'
                    })
            except Exception as e:
                processing_results['failed'].append({
                    'file': filename,
                    'reason': str(e)
                })
                logger.error(f"OCR error for {filename}: {e}")
        
        if not session.ocr_texts:
            return jsonify({
                'error': 'OCR processing failed for all files',
                'details': processing_results
            }), 500
        
        # Step 2: Create Vector DB (RAG)
        logger.info(f"Creating vector database for {customer_id}")
        if not session.create_vector_db():
            return jsonify({'error': 'Vector DB creation failed'}), 500
        
        # Step 3: Retrieve context
        logger.info(f"Retrieving context for {customer_id}")
        query = "Complete financial profile including income, expenses, assets, liabilities, and repayment ability"
        docs = session.vector_db.similarity_search(query, k=5)
        context = "\n".join([d.page_content for d in docs])
        
        # Step 4: Comprehensive Analysis
        logger.info(f"Generating comprehensive analysis for {customer_id}")
        risk_analysis = analyze_multi_document_risk(context, customer_id, session.files.keys())
        
        # Step 5: Extract Financial Metrics
        financial_metrics = extract_financial_metrics(context)
        
        # Step 6: Generate Report
        report = generate_customer_report(session)
        
        session.analysis_results = {
            'risk_analysis': risk_analysis,
            'financial_metrics': financial_metrics,
            'report': report,
            'processed_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'customer_id': customer_id,
            'processing_results': processing_results,
            'ocr_summary': {
                'files_processed': len(session.ocr_texts),
                'total_text_extracted': len(session.all_text)
            },
            'analysis': {
                'risk_assessment': risk_analysis[:1000] + "..." if len(str(risk_analysis)) > 1000 else risk_analysis,
                'financial_metrics': financial_metrics[:500] + "..." if len(str(financial_metrics)) > 500 else financial_metrics,
                'has_full_report': True
            },
            'message': 'All documents processed and analyzed successfully'
        }), 200
    
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-analysis', methods=['POST'])
def get_analysis():
    """Get the complete analysis for a customer"""
    try:
        data = request.json
        customer_id = data.get('customer_id')
        
        if not customer_id or customer_id not in customer_sessions:
            return jsonify({'error': 'Customer session not found'}), 404
        
        session = customer_sessions[customer_id]
        
        if not session.analysis_results:
            return jsonify({'error': 'No analysis available. Process documents first.'}), 400
        
        return jsonify({
            'success': True,
            'customer_id': customer_id,
            'session_info': session.get_session_info(),
            'analysis': session.analysis_results
        }), 200
    
    except Exception as e:
        logger.error(f"Retrieval Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/customer-summary', methods=['POST'])
def customer_summary():
    """Get a summary of a customer's session"""
    try:
        data = request.json
        customer_id = data.get('customer_id')
        
        if not customer_id or customer_id not in customer_sessions:
            return jsonify({'error': 'Customer session not found'}), 404
        
        session = customer_sessions[customer_id]
        
        return jsonify({
            'success': True,
            'session': session.get_session_info(),
            'has_analysis': bool(session.analysis_results)
        }), 200
    
    except Exception as e:
        logger.error(f"Summary Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch-documents', methods=['POST'])
def fetch_documents_from_apis():
    """Fetch financial documents from APIs and add to session (integrated)"""
    try:
        from financial_api_integration import FinancialDataIntegration
        
        data = request.json
        customer_id = data.get('customer_id')
        
        if not customer_id:
            return jsonify({'error': 'Customer ID required'}), 400
        
        if customer_id not in customer_sessions:
            return jsonify({'error': 'Customer session not found'}), 404
        
        session = customer_sessions[customer_id]
        
        # Prepare customer data for API calls
        customer_data = {
            'pan_number': data.get('pan_number'),
            'gst_number': data.get('gst_number'),
            'bank_account': {
                'account_number': data.get('account_number'),
                'ifsc_code': data.get('ifsc_code')
            } if data.get('account_number') else None,
            'cin_number': data.get('cin_number'),
            'months': data.get('months', 6)
        }
        
        # Filter out None values
        customer_data = {k: v for k, v in customer_data.items() if v is not None}
        
        logger.info(f"Fetching API documents for customer {customer_id}")
        
        # Initialize API integration
        api_integration = FinancialDataIntegration()
        
        # Fetch all available documents
        api_results = api_integration.fetch_all_documents(customer_data)
        
        # Add each successfully fetched document to session
        for doc_type, doc_data in api_results['data'].items():
            try:
                session.add_api_data(doc_type, doc_data)
                logger.info(f"✓ Added {doc_type} from API to session {customer_id}")
            except Exception as e:
                logger.error(f"Error adding {doc_type} to session: {e}")
                api_results['failed'].append(f"{doc_type}: {str(e)}")
        
        return jsonify({
            'success': True,
            'customer_id': customer_id,
            'successful_documents': api_results['success'],
            'failed_documents': api_results['failed'],
            'total_documents_fetched': len(api_results['success']),
            'total_text_added': len(session.all_text),
            'api_sources_count': len(session.data_sources['api_sources']),
            'message': f'Successfully fetched and integrated {len(api_results["success"])} API documents'
        }), 200
    
    except ImportError:
        logger.warning("financial_api_integration not available")
        return jsonify({'error': 'API integration module not available'}), 503
    except Exception as e:
        logger.error(f"API Fetch Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch-gst', methods=['POST'])
def fetch_gst_data():
    """Fetch GST details and auto-add to session"""
    try:
        from financial_api_integration import GSTApi
        
        data = request.json
        gst_number = data.get('gst_number')
        customer_id = data.get('customer_id')
        
        if not gst_number:
            return jsonify({'error': 'GST number required'}), 400
        
        gst_api = GSTApi()
        
        logger.info(f"Fetching GST data for {gst_number[:8]}****")
        gst_data = gst_api.fetch_gst_details(gst_number)
        
        # If customer session exists, add the data to session
        if customer_id and customer_id in customer_sessions:
            session = customer_sessions[customer_id]
            session.add_api_data('gst', gst_data)
            logger.info(f"✓ GST data added to session {customer_id}")
            return jsonify({
                'success': True,
                'gst_data': gst_data,
                'added_to_session': True,
                'message': 'GST data fetched and added to session'
            }), 200
        else:
            return jsonify({
                'success': True,
                'gst_data': gst_data,
                'added_to_session': False,
                'message': 'GST data fetched. Create session to add to profile.'
            }), 200
    
    except ImportError:
        return jsonify({'error': 'API integration module not available'}), 503
    except Exception as e:
        logger.error(f"GST Fetch Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch-itr', methods=['POST'])
def fetch_itr_data():
    """Fetch ITR details and auto-add to session"""
    try:
        from financial_api_integration import ITRApi
        
        data = request.json
        pan_number = data.get('pan_number')
        customer_id = data.get('customer_id')
        years = data.get('years', 3)
        
        if not pan_number:
            return jsonify({'error': 'PAN number required'}), 400
        
        itr_api = ITRApi()
        
        logger.info(f"Fetching ITR data for {pan_number} ({years} years)")
        itr_data = itr_api.fetch_itr_details(pan_number, years)
        
        # If customer session exists, add the data to session
        if customer_id and customer_id in customer_sessions:
            session = customer_sessions[customer_id]
            session.add_api_data('itr', itr_data)
            logger.info(f"✓ ITR data added to session {customer_id}")
            return jsonify({
                'success': True,
                'itr_data': itr_data,
                'added_to_session': True,
                'message': f'ITR data fetched ({years} years) and added to session'
            }), 200
        else:
            return jsonify({
                'success': True,
                'itr_data': itr_data,
                'added_to_session': False,
                'message': 'ITR data fetched. Create session to add to profile.'
            }), 200
    
    except ImportError:
        return jsonify({'error': 'API integration module not available'}), 503
    except Exception as e:
        logger.error(f"ITR Fetch Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch-bank-statement', methods=['POST'])
def fetch_bank_statement_data():
    """Fetch bank statement and auto-add to session"""
    try:
        from financial_api_integration import BankStatementAPI
        
        data = request.json
        account_number = data.get('account_number')
        ifsc_code = data.get('ifsc_code')
        customer_id = data.get('customer_id')
        months = data.get('months', 6)
        
        if not account_number or not ifsc_code:
            return jsonify({'error': 'Account number and IFSC code required'}), 400
        
        bank_api = BankStatementAPI()
        
        account_info = {
            'account_number': account_number,
            'ifsc_code': ifsc_code
        }
        
        logger.info(f"Fetching bank statement for {account_number[-4:]}**** ({months} months)")
        bank_data = bank_api.fetch_bank_statement(account_info, months)
        
        # If customer session exists, add the data to session
        if customer_id and customer_id in customer_sessions:
            session = customer_sessions[customer_id]
            session.add_api_data('bank_statement', bank_data)
            logger.info(f"✓ Bank statement added to session {customer_id}")
            return jsonify({
                'success': True,
                'bank_data': bank_data,
                'added_to_session': True,
                'message': f'Bank statement fetched ({months} months) and added to session'
            }), 200
        else:
            return jsonify({
                'success': True,
                'bank_data': bank_data,
                'added_to_session': False,
                'message': 'Bank statement fetched. Create session to add to profile.'
            }), 200
    
    except ImportError:
        return jsonify({'error': 'API integration module not available'}), 503
    except Exception as e:
        logger.error(f"Bank Statement Fetch Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch-credit-report', methods=['POST'])
def fetch_credit_report_data():
    """Fetch credit report and auto-add to session"""
    try:
        from financial_api_integration import CreditBureauAPI
        
        data = request.json
        pan_number = data.get('pan_number')
        customer_id = data.get('customer_id')
        
        if not pan_number:
            return jsonify({'error': 'PAN number required'}), 400
        
        credit_api = CreditBureauAPI()
        
        logger.info(f"Fetching credit report for {pan_number}")
        credit_data = credit_api.fetch_credit_report(pan_number)
        
        # If customer session exists, add the data to session
        if customer_id and customer_id in customer_sessions:
            session = customer_sessions[customer_id]
            session.add_api_data('credit_report', credit_data)
            logger.info(f"✓ Credit report added to session {customer_id}")
            return jsonify({
                'success': True,
                'credit_data': credit_data,
                'added_to_session': True,
                'message': 'Credit report fetched and added to session'
            }), 200
        else:
            return jsonify({
                'success': True,
                'credit_data': credit_data,
                'added_to_session': False,
                'message': 'Credit report fetched. Create session to add to profile.'
            }), 200
    
    except ImportError:
        return jsonify({'error': 'API integration module not available'}), 503
    except Exception as e:
        logger.error(f"Credit Report Fetch Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch-company-info', methods=['POST'])
def fetch_company_info_data():
    """Fetch company information and auto-add to session"""
    try:
        from financial_api_integration import CompanyInfoAPI
        
        data = request.json
        cin_number = data.get('cin_number')
        customer_id = data.get('customer_id')
        
        if not cin_number:
            return jsonify({'error': 'CIN number required'}), 400
        
        company_api = CompanyInfoAPI()
        
        logger.info(f"Fetching company info for {cin_number}")
        company_data = company_api.fetch_company_details(cin_number)
        
        # If customer session exists, add the data to session
        if customer_id and customer_id in customer_sessions:
            session = customer_sessions[customer_id]
            session.add_api_data('company_info', company_data)
            logger.info(f"✓ Company info added to session {customer_id}")
            return jsonify({
                'success': True,
                'company_data': company_data,
                'added_to_session': True,
                'message': 'Company information fetched and added to session'
            }), 200
        else:
            return jsonify({
                'success': True,
                'company_data': company_data,
                'added_to_session': False,
                'message': 'Company information fetched. Create session to add to profile.'
            }), 200
    
    except ImportError:
        return jsonify({'error': 'API integration module not available'}), 503
    except Exception as e:
        logger.error(f"Company Info Fetch Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
