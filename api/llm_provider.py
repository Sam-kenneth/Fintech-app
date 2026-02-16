"""
LLM Provider Module
Supports: Grok, Mistral, Groq APIs (no local Ollama needed)
"""

import os
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GrokLLM:
    """Grok API LLM Provider"""
    
    def __init__(self):
        self.api_key = os.getenv('GROK_API_KEY')
        self.model = os.getenv('GROK_MODEL', 'grok-beta')
        self.base_url = "https://api.xai-1.ai/v1"
        
        if not self.api_key:
            logger.warning("GROK_API_KEY not set")
    
    def generate_analysis(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate analysis using Grok API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a fintech credit risk analyst. Provide detailed, professional analysis.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.7,
                'max_tokens': max_tokens
            }
            
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                error_msg = f"Grok API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
        
        except requests.exceptions.Timeout:
            raise Exception("Grok API timeout")
        except Exception as e:
            logger.error(f"Grok generation error: {e}")
            raise


class MistralLLM:
    """Mistral AI LLM Provider"""
    
    def __init__(self):
        self.api_key = os.getenv('MISTRAL_API_KEY')
        self.model = os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
        self.base_url = "https://api.mistral.ai/v1"
        
        if not self.api_key:
            logger.warning("MISTRAL_API_KEY not set")
    
    def generate_analysis(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate analysis using Mistral API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a fintech credit risk analyst. Provide detailed, professional analysis.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.7,
                'max_tokens': max_tokens
            }
            
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                error_msg = f"Mistral API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
        
        except requests.exceptions.Timeout:
            raise Exception("Mistral API timeout")
        except Exception as e:
            logger.error(f"Mistral generation error: {e}")
            raise


class GroqLLM:
    """Groq Cloud LLM Provider (Fastest Inference)"""
    
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.model = os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')
        self.base_url = "https://api.groq.com/openai/v1"
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set")
    
    def generate_analysis(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate analysis using Groq API (Fastest)"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a fintech credit risk analyst. Provide detailed, professional analysis.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.7,
                'max_tokens': max_tokens
            }
            
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                error_msg = f"Groq API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
        
        except requests.exceptions.Timeout:
            raise Exception("Groq API timeout")
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            raise


def get_llm_provider():
    """
    Factory function to get the appropriate LLM provider
    Based on LLM_PROVIDER environment variable
    
    Options: 'grok', 'mistral', 'groq'
    Default: 'grok'
    """
    provider = os.getenv('LLM_PROVIDER', 'grok').lower()
    
    if provider == 'mistral':
        logger.info("Using Mistral LLM provider")
        return MistralLLM()
    elif provider == 'groq':
        logger.info("Using Groq LLM provider")
        return GroqLLM()
    else:  # Default to Grok
        logger.info("Using Grok LLM provider (default)")
        return GrokLLM()


def generate_risk_analysis(context: str, customer_id: str, file_list: list) -> str:
    """
    Generate comprehensive risk assessment using the selected LLM provider
    """
    llm = get_llm_provider()
    
    files_str = "\n".join([f"- {f}" for f in file_list])
    
    prompt = f"""
You are a fintech credit risk analyst reviewing a customer's financial profile.

CUSTOMER ID: {customer_id}
DOCUMENTS ANALYZED:
{files_str}

EXTRACTED CONTEXT FROM ALL DOCUMENTS:
{context[:3000]}

TASK: Provide a comprehensive risk assessment based on ALL available documents.

Structure your response as follows:

1. CUSTOMER PROFILE
   - Brief summary of the customer based on documents

2. RISK SCORE
   - Overall risk score (0-100, where 100 is highest risk)
   - Risk category: Low / Moderate / High / Critical

3. INCOME ANALYSIS
   - Income stability and consistency
   - Income sources identified
   - Income verification level

4. PAYMENT HISTORY
   - EMI payment patterns
   - Payment discipline assessment
   - Any missed payments or defaults

5. DEBT ANALYSIS
   - Total outstanding debt
   - Debt-to-income ratio (if available)
   - Debt repayment capacity

6. RED FLAGS
   - List any concerning patterns
   - Risk factors identified
   - Negative indicators

7. POSITIVE FACTORS
   - List positive indicators
   - Strengths in the profile
   - Mitigating factors

8. MISSING DOCUMENTS
   - Any important documents that would strengthen assessment
   - Recommendations for additional verification

9. RECOMMENDATION
   - Lending recommendation (Approve/Reject/Conditional)
   - Suggested loan amount (if applicable)
   - Recommended interest rate (if applicable)
   - Conditions or requirements

10. CONFIDENCE LEVEL
    - Confidence in assessment (0-100%)
    - Factors affecting confidence

Provide clear, professional analysis suitable for credit decision making.
"""
    
    return llm.generate_analysis(prompt)


def generate_financial_metrics(context: str) -> str:
    """
    Extract and analyze financial metrics from documents
    """
    llm = get_llm_provider()
    
    prompt = f"""
Extract and analyze key financial metrics from the following financial documents:

{context[:2000]}

Provide analysis of:
1. Monthly/Annual income (with sources)
2. Total outstanding debt
3. Debt obligations breakdown
4. Credit score (if mentioned)
5. Employment status
6. Available savings/assets
7. Loan amount requested (if mentioned)
8. Loan tenure requested (if mentioned)
9. Debt-to-Income ratio
10. Monthly expense estimates

Format as clear, structured analysis suitable for financial assessment.
"""
    
    return llm.generate_analysis(prompt)


def generate_professional_report(context: str, customer_id: str) -> str:
    """
    Generate a professional credit assessment report
    """
    llm = get_llm_provider()
    
    prompt = f"""
Generate a professional credit assessment report for loan application.

CUSTOMER ID: {customer_id}
ASSESSMENT DATE: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DOCUMENT SUMMARY:
{context[:2500]}

Create a formal report with these sections:

EXECUTIVE SUMMARY
- Brief overall assessment
- Key recommendation
- Risk level

APPLICANT PROFILE
- Applicant details
- Employment/business background
- Loan purpose

FINANCIAL ANALYSIS
- Income analysis
- Debt analysis
- Financial stability assessment
- Liquidity position

CREDIT RISK ASSESSMENT
- Overall risk rating
- Key risk factors
- Mitigating factors
- Credit behavior assessment

KEY OBSERVATIONS
- Notable findings
- Patterns observed
- Concerns identified
- Strengths noted

FINAL RECOMMENDATION
- Lending decision
- Loan amount recommendation
- Tenure suggestion
- Interest rate guidance
- Terms and conditions
- Any special requirements

Format as a professional financial report suitable for:
- Credit approval committee
- Customer presentation
- Internal documentation
- Risk management review

Use clear, professional language throughout.
"""
    
    return llm.generate_analysis(prompt)


if __name__ == '__main__':
    # Test the module
    print("Testing LLM providers...")
    
    try:
        provider = get_llm_provider()
        print(f"✓ LLM provider initialized: {provider.__class__.__name__}")
        
        # Test generation
        test_prompt = "Provide a brief summary of fintech risk assessment."
        result = provider.generate_analysis(test_prompt, max_tokens=500)
        print(f"✓ Test generation successful")
        print(f"Response length: {len(result)} characters")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure to set the appropriate API key in environment variables:")
        print("- GROK_API_KEY for Grok")
        print("- MISTRAL_API_KEY for Mistral")
        print("- GROQ_API_KEY for Groq")
