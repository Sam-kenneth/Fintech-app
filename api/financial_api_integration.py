"""
Financial Data API Integrations Module
Integrates with multiple APIs to fetch GST, ITR, Bank Statements, etc.
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import base64

logger = logging.getLogger(__name__)

class FinancialDataIntegration:
    """Main class for integrating multiple financial data APIs"""
    
    def __init__(self):
        # API Configuration (add your API keys in environment variables)
        self.apis = {
            'pancard': PancardAPI(),
            'gst': GSTApi(),
            'itr': ITRApi(),
            'bank_statement': BankStatementAPI(),
            'credit_bureau': CreditBureauAPI(),
            'company_info': CompanyInfoAPI(),
        }
    
    def fetch_all_documents(self, customer_data: Dict) -> Dict:
        """Fetch all available documents for a customer"""
        results = {
            'success': [],
            'failed': [],
            'data': {}
        }
        
        # Fetch PAN details
        if customer_data.get('pan_number'):
            try:
                pan_data = self.apis['pancard'].fetch_pan_details(customer_data['pan_number'])
                results['data']['pan'] = pan_data
                results['success'].append('PAN Details')
            except Exception as e:
                results['failed'].append(f"PAN: {str(e)}")
        
        # Fetch GST details
        if customer_data.get('gst_number'):
            try:
                gst_data = self.apis['gst'].fetch_gst_details(customer_data['gst_number'])
                results['data']['gst'] = gst_data
                results['success'].append('GST Details')
            except Exception as e:
                results['failed'].append(f"GST: {str(e)}")
        
        # Fetch ITR details
        if customer_data.get('pan_number'):
            try:
                itr_data = self.apis['itr'].fetch_itr_details(customer_data['pan_number'])
                results['data']['itr'] = itr_data
                results['success'].append('ITR Details')
            except Exception as e:
                results['failed'].append(f"ITR: {str(e)}")
        
        # Fetch Bank Statements
        if customer_data.get('bank_account'):
            try:
                bank_data = self.apis['bank_statement'].fetch_bank_statement(
                    customer_data['bank_account'],
                    customer_data.get('months', 6)
                )
                results['data']['bank_statement'] = bank_data
                results['success'].append('Bank Statement')
            except Exception as e:
                results['failed'].append(f"Bank Statement: {str(e)}")
        
        # Fetch Credit Report
        if customer_data.get('pan_number'):
            try:
                credit_data = self.apis['credit_bureau'].fetch_credit_report(
                    customer_data['pan_number']
                )
                results['data']['credit_report'] = credit_data
                results['success'].append('Credit Report')
            except Exception as e:
                results['failed'].append(f"Credit Report: {str(e)}")
        
        # Fetch Company Info (if applicable)
        if customer_data.get('cin_number'):
            try:
                company_data = self.apis['company_info'].fetch_company_details(
                    customer_data['cin_number']
                )
                results['data']['company_info'] = company_data
                results['success'].append('Company Info')
            except Exception as e:
                results['failed'].append(f"Company Info: {str(e)}")
        
        return results


class PancardAPI:
    """Integration with PAN verification APIs"""
    
    def __init__(self):
        # Use services like:
        # - Postman PAN Verification API
        # - eway.io
        # - InstantKYC
        self.base_url = "https://api.example.com/pan"
        self.api_key = "your_pan_api_key"
    
    def fetch_pan_details(self, pan_number: str) -> Dict:
        """Fetch PAN details from verification service"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'pan': pan_number.upper()
            }
            
            # Example API call
            response = requests.post(
                f"{self.base_url}/verify",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'pan_number': pan_number,
                    'name': data.get('name'),
                    'father_name': data.get('father_name'),
                    'dob': data.get('dob'),
                    'pan_status': data.get('status'),
                    'aadhaar_linked': data.get('aadhaar_linked'),
                    'verified': True,
                    'fetch_date': datetime.now().isoformat()
                }
            else:
                raise Exception(f"PAN API Error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"PAN fetch error: {e}")
            raise


class GSTApi:
    """Integration with GST APIs"""
    
    def __init__(self):
        # Use services like:
        # - GST Tracking System API
        # - GSTIN Verification APIs
        # - Vayana
        self.base_url = "https://api.example.com/gst"
        self.api_key = "your_gst_api_key"
    
    def fetch_gst_details(self, gst_number: str) -> Dict:
        """Fetch GST registration details"""
        try:
            headers = {
                'x-api-key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            # Example API call
            response = requests.get(
                f"{self.base_url}/search/{gst_number}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'gstin': gst_number,
                    'business_name': data.get('business_name'),
                    'registration_date': data.get('registration_date'),
                    'status': data.get('status'),
                    'address': data.get('address'),
                    'turnover': data.get('turnover'),
                    'principal_place': data.get('principal_place'),
                    'business_activities': data.get('business_activities'),
                    'bank_details': data.get('bank_details'),
                    'fetch_date': datetime.now().isoformat()
                }
            else:
                raise Exception(f"GST API Error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"GST fetch error: {e}")
            raise


class ITRApi:
    """Integration with Income Tax Return APIs"""
    
    def __init__(self):
        # Use services like:
        # - Income Tax e-filing APIs
        # - Tax portal integrations
        # - Vayana ITR APIs
        self.base_url = "https://api.example.com/itr"
        self.api_key = "your_itr_api_key"
    
    def fetch_itr_details(self, pan_number: str, years: int = 3) -> Dict:
        """Fetch ITR details for specified years"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            itr_data = {
                'pan': pan_number,
                'itr_records': []
            }
            
            # Fetch last N years of ITR
            for i in range(years):
                year = datetime.now().year - i
                payload = {
                    'pan': pan_number.upper(),
                    'assessment_year': year
                }
                
                response = requests.post(
                    f"{self.base_url}/fetch",
                    json=payload,
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    record = response.json()
                    itr_data['itr_records'].append({
                        'assessment_year': year,
                        'gross_income': record.get('gross_income'),
                        'taxable_income': record.get('taxable_income'),
                        'tax_paid': record.get('tax_paid'),
                        'income_sources': record.get('income_sources'),
                        'deductions': record.get('deductions'),
                        'filing_date': record.get('filing_date'),
                        'status': record.get('status')
                    })
            
            itr_data['fetch_date'] = datetime.now().isoformat()
            return itr_data
        
        except Exception as e:
            logger.error(f"ITR fetch error: {e}")
            raise


class BankStatementAPI:
    """Integration with Bank APIs for statement retrieval"""
    
    def __init__(self):
        # Use services like:
        # - Open Banking APIs (RBI standards)
        # - Yodlee
        # - Finicity
        # - Plaid
        # - Individual bank APIs (ICICI, HDFC, SBI, etc.)
        self.base_url = "https://api.example.com/banking"
        self.api_key = "your_banking_api_key"
    
    def fetch_bank_statement(self, account_info: Dict, months: int = 6) -> Dict:
        """Fetch bank statements via API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30*months)
            
            payload = {
                'account_number': account_info.get('account_number'),
                'ifsc_code': account_info.get('ifsc_code'),
                'from_date': start_date.strftime('%Y-%m-%d'),
                'to_date': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.post(
                f"{self.base_url}/statement",
                json=payload,
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'account_number': account_info.get('account_number'),
                    'bank_name': data.get('bank_name'),
                    'account_holder': data.get('account_holder'),
                    'opening_balance': data.get('opening_balance'),
                    'closing_balance': data.get('closing_balance'),
                    'total_credits': data.get('total_credits'),
                    'total_debits': data.get('total_debits'),
                    'average_balance': data.get('average_balance'),
                    'transactions': data.get('transactions', []),
                    'period_from': start_date.isoformat(),
                    'period_to': end_date.isoformat(),
                    'fetch_date': datetime.now().isoformat()
                }
            else:
                raise Exception(f"Banking API Error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Bank statement fetch error: {e}")
            raise


class CreditBureauAPI:
    """Integration with Credit Bureau APIs"""
    
    def __init__(self):
        # Use services like:
        # - CIBIL (Equifax)
        # - Experian
        # - Crif High Mark
        # - SMCB
        self.base_url = "https://api.example.com/credit"
        self.api_key = "your_credit_api_key"
    
    def fetch_credit_report(self, pan_number: str) -> Dict:
        """Fetch credit report from credit bureau"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'pan': pan_number.upper()
            }
            
            response = requests.post(
                f"{self.base_url}/report",
                json=payload,
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'pan': pan_number,
                    'credit_score': data.get('credit_score'),
                    'credit_rating': data.get('credit_rating'),
                    'total_accounts': data.get('total_accounts'),
                    'active_accounts': data.get('active_accounts'),
                    'closed_accounts': data.get('closed_accounts'),
                    'delinquent_accounts': data.get('delinquent_accounts'),
                    'enquiries_last_6_months': data.get('enquiries_6m'),
                    'enquiries_last_12_months': data.get('enquiries_12m'),
                    'default_history': data.get('default_history'),
                    'loan_details': data.get('loan_details'),
                    'credit_card_details': data.get('credit_card_details'),
                    'fetch_date': datetime.now().isoformat()
                }
            else:
                raise Exception(f"Credit Bureau API Error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Credit report fetch error: {e}")
            raise


class CompanyInfoAPI:
    """Integration with Company Registration APIs"""
    
    def __init__(self):
        # Use services like:
        # - MCA (Ministry of Corporate Affairs)
        # - Crunchbase
        # - Apollo
        # - Clearbit
        self.base_url = "https://api.example.com/company"
        self.api_key = "your_company_api_key"
    
    def fetch_company_details(self, cin_number: str) -> Dict:
        """Fetch company details from MCA/registrar"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'cin': cin_number.upper()
            }
            
            response = requests.post(
                f"{self.base_url}/details",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'cin': cin_number,
                    'company_name': data.get('company_name'),
                    'registration_date': data.get('registration_date'),
                    'status': data.get('status'),
                    'registered_office': data.get('registered_office'),
                    'directors': data.get('directors'),
                    'authorized_capital': data.get('authorized_capital'),
                    'paid_up_capital': data.get('paid_up_capital'),
                    'annual_returns': data.get('annual_returns'),
                    'financial_summary': data.get('financial_summary'),
                    'fetch_date': datetime.now().isoformat()
                }
            else:
                raise Exception(f"Company API Error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Company info fetch error: {e}")
            raise


# Helper function to convert API data to text format
def api_data_to_text(api_data: Dict, doc_type: str) -> str:
    """Convert API response data to text for document processing"""
    
    text = f"\n{'='*80}\n"
    text += f"DOCUMENT: {doc_type.upper()}\n"
    text += f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    text += f"{'='*80}\n\n"
    
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
    
    text += dict_to_text(api_data)
    return text
