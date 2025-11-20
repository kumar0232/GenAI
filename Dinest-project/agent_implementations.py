import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from groq import Groq
from sentence_transformers import SentenceTransformer
import requests
from supabase import create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LLMUtils:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Initialized LLM interface with Groq and SentenceTransformers")

    def generate_response(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        try:
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            messages.append({"role": "user", "content": prompt})
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating Groq response: {e}")
            return f"I encountered an error while processing your request. Error: {str(e)}"

    def get_embedding(self, text: str) -> List[float]:
        return self.embedder.encode(text).tolist()

class BaseAgent:
    def __init__(self, llm_utils: LLMUtils):
        self.llm_utils = llm_utils

    def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        raise NotImplementedError("Subclasses must implement this method")

class RouterAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils):
        super().__init__(llm_utils)
        self.system_prompt = """
        You are a Router Agent for TechSolutions customer support. Your job is to:
        1. Understand the customer's query
        2. Classify the query into one of these categories:
           - Product: Questions about products, features, pricing, plans
           - Technical: Questions about errors, issues, troubleshooting
           - Billing: Questions about orders, invoices, payments, subscriptions
           - Account: Questions about user management, access, settings
           - General: General inquiries that don't fit other categories
        3. For multi-part queries, identify each part and its category
        
        Respond with JSON in this format:
        {
            "classification": "Product|Technical|Billing|Account|General",
            "confidence": 0.9,
            "requires_clarification": false,
            "clarification_question": ""
        }
        For multi-part queries:
        {
            "multi_part": true,
            "parts": [
                {
                    "query_part": "text",
                    "classification": "Product|Technical|Billing|Account|General"
                }
            ]
        }
        """

    def process(self, query: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        prompt = f"CLASSIFY QUERY: {query}\n\nOUTPUT JSON:"
        response = self.llm_utils.generate_response(prompt, self.system_prompt)
        try:
            cleaned_response = self._clean_json_response(response)
            result = json.loads(cleaned_response)
            return self._validate_response_structure(result)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed. Raw response: {response}")
            return self._safe_fallback_response(query, e)

    def _clean_json_response(self, response: str) -> str:
        response = response.replace('```json', '').replace('```', '')
        start = response.find('{')
        end = response.rfind('}') + 1
        return response[start:end] if start != -1 and end != 0 else response

    def _validate_response_structure(self, result: Dict) -> Dict:
        required_single = ["classification", "confidence", "requires_clarification"]
        required_multi = ["multi_part", "parts"]
        if "multi_part" in result:
            if not all(k in result for k in required_multi):
                raise ValueError("Invalid multi-part structure")
            for part in result.get("parts", []):
                if "query_part" not in part or "classification" not in part:
                    raise ValueError("Invalid part structure")
        else:
            if not all(k in result for k in required_single):
                raise ValueError("Missing required fields")
        return result

    def _safe_fallback_response(self, query: str, error: Exception) -> Dict:
        logger.warning(f"Using fallback classification for query: {query}")
        return {
            "classification": "General",
            "confidence": 0.5,
            "requires_clarification": True,
            "clarification_question": "Could you please rephrase or provide more details about your question?",
            "parse_error": str(error)
        }

class ProductSpecialistAgent(BaseAgent):
    def __init__(
        self,
        llm_utils: LLMUtils,
        product_catalog: Dict[str, Any],
        faqs: Dict[str, Any],
        vector_db: str,
    ):
        super().__init__(llm_utils)
        self.product_catalog = product_catalog
        self.faqs = faqs
        self.vector_db = vector_db
        self.supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        self.system_prompt = """
        You are a Product Specialist Agent for TechSolutions customer support.
        You're an expert on TechSolutions products, features, pricing, and plans.
        When responding to customer queries:
        1. Be accurate and specific about product features and pricing
        2. Compare products when relevant to help customers choose
        3. Highlight benefits and use cases for specific products
        4. If you don't know something, say so rather than guessing
        Keep your responses friendly, concise, and focused on answering the customer's specific question.
        """

    def _retrieve_relevant_information(self, query: str) -> str:
        try:
            embedding = self.llm_utils.get_embedding(query)
            response = self.supabase.rpc(
                "match_documents",
                {"query_embedding": embedding, "match_count": 3},
                table=self.vector_db
            ).execute()
            return "\n\n".join([doc["content"] for doc in response.data])
        except Exception as e:
            logger.error(f"Error retrieving information from Supabase: {e}")
            return ""

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        relevant_info = self._retrieve_relevant_information(query)
        prompt = f"""
        Customer query: {query}
        Relevant information:
        {relevant_info}
        Please provide a helpful response based on this information.
        """
        return self.llm_utils.generate_response(prompt, self.system_prompt)

class TechnicalSupportAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils, tech_docs: str, vector_db: str):
        super().__init__(llm_utils)
        self.tech_docs = tech_docs
        self.vector_db = vector_db
        self.supabase = create_client(os.getenv("SUPABASE_URL","https://pkdcrqdwqyfrjenuwcls.supabase.co" ), os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBrZGNycWR3cXlmcmplbnV3Y2xzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Njk2MjgzOSwiZXhwIjoyMDYyNTM4ODM5fQ.6BPvnnCp6ctiYrRwnbb5muGVHnIP1A3fEq5_teDNKfE"))
        self.system_prompt = """
        You are a Technical Support Agent for TechSolutions customer support.
        You're an expert in troubleshooting TechSolutions products and resolving technical issues.
        When responding to customer queries:
        1. Identify the specific issue or error described
        2. Provide step-by-step troubleshooting instructions
        3. Reference relevant documentation when applicable
        4. Suggest preventive measures for future reference
        Keep your responses clear, structured, and focused on resolving the customer's technical problem.
        """

    def _retrieve_troubleshooting_info(self, query: str) -> str:
        try:
            embedding = self.llm_utils.get_embedding(query)
            response = self.supabase.rpc(
                "match_documents",
                {"query_embedding": embedding, "match_count": 3},
                table=self.vector_db
            ).execute()
            return "\n\n".join([doc["content"] for doc in response.data])
        except Exception as e:
            logger.error(f"Error retrieving troubleshooting information: {e}")
            return ""

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        relevant_info = self._retrieve_troubleshooting_info(query)
        diagnostic_info = await self._call_diagnostic_api(query)
        diagnostic_text = ""
        if diagnostic_info:
            solutions = diagnostic_info.get("solutions", [])
            solutions_text = "\n".join([f"- {solution}" for solution in solutions])
            diagnostic_text = f"""
            Diagnostic results:
            Issue: {diagnostic_info.get('name', 'Unknown issue')}
            Suggested solutions:
            {solutions_text}
            Documentation: {diagnostic_info.get('documentation_link', '')}
            """
        prompt = f"""
        Customer query: {query}
        Relevant troubleshooting information:
        {relevant_info}
        {diagnostic_text}
        Please provide a helpful response to resolve this technical issue.
        """
        return self.llm_utils.generate_response(prompt, self.system_prompt)

class OrderBillingAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils, product_catalog: Dict[str, Any]):
        super().__init__(llm_utils)
        self.product_catalog = product_catalog
        self.system_prompt = """
        You are an Order and Billing Agent for TechSolutions customer support.
        You're an expert in handling inquiries about orders, invoices, payments, and subscriptions.
        When responding to customer queries:
        1. Be precise about order status, payment information, and subscription details
        2. Explain billing charges clearly and transparently
        3. Outline available payment options and subscription changes when relevant
        4. Maintain a professional and reassuring tone
        Keep your responses clear, specific, and focused on addressing the customer's billing-related questions.
        """

    async def _get_order_details(self, order_id: str) -> Dict[str, Any]:
        try:
            response = await asyncio.to_thread(
                requests.get, f"http://localhost:8000/api/orders/{order_id}"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Order not found: {order_id}")
                return {}
            else:
                logger.error(f"Error retrieving order details: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving order details: {e}")
            return {}

    async def _get_account_details(self, account_id: str) -> Dict[str, Any]:
        try:
            response = await asyncio.to_thread(
                requests.get, f"http://localhost:8000/api/accounts/{account_id}"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Account not found: {account_id}")
                return {}
            else:
                logger.error(f"Error retrieving account details: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving account details: {e}")
            return {}

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        import re
        order_id = None
        order_match = re.search(r"ORD-\d+", query)
        if order_match:
            order_id = order_match.group(0)
        account_id = None
        account_match = re.search(r"ACC-\d+", query)
        if account_match:
            account_id = account_match.group(0)
        order_details = await self._get_order_details(order_id) if order_id else {}
        account_details = await self._get_account_details(account_id) if account_id else {}
        prompt = f"Customer query: {query}\n\n"
        if order_details:
            prompt += f"Order information:\n{json.dumps(order_details, indent=2)}\n\n"
        if account_details:
            prompt += f"Account information:\n{json.dumps(account_details, indent=2)}\n\n"
        if "pricing" in query.lower() or "cost" in query.lower() or "price" in query.lower():
            products_info = json.dumps(self.product_catalog.get("products", []), indent=2)
            prompt += f"Product pricing information:\n{products_info}\n\n"
        prompt += "Please provide a helpful response to this billing or order question."
        return self.llm_utils.generate_response(prompt, self.system_prompt)

class AccountManagementAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils):
        super().__init__(llm_utils)
        self.system_prompt = """
        You are an Account Management Agent for TechSolutions customer support.
        You're an expert in handling account queries, including user management, subscription details, and available user slots.
        When responding to customer queries:
        1. Confirm the action requested
        2. Retrieve account details to check the current subscription tier and available user slots
        3. Provide step-by-step instructions for requested actions
        4. Offer additional suggestions if the account has reached its user limit
        Keep your responses clear, structured, and detailed.
        """

    async def _get_account_info(self, account_id: str) -> Dict[str, Any]:
        try:
            response = await asyncio.to_thread(
                requests.get, f"http://localhost:8000/api/accounts/{account_id}"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Account not found: {account_id}")
                return {}
            else:
                logger.error(f"Error retrieving account details: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving account details: {e}")
            return {}

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        import re
        account_id = None
        account_match = re.search(r"ACC-\d+", query)
        if account_match:
            account_id = account_match.group(0)
        if not account_id:
            account_id = "ACC-1111"
        account_info = await self._get_account_info(account_id)
        subscription = account_info.get("subscription", {})
        plan = subscription.get("plan", "unknown")
        if plan.lower() == "cm-pro":
            user_limit = 20
        elif plan.lower() == "cm-enterprise":
            user_limit = float("inf")
        else:
            user_limit = 5
        current_user_count = len(account_info.get("users", []))
        available_slots = "unlimited" if user_limit == float("inf") else max(0, user_limit - current_user_count)
        prompt = f"""
        Customer query: {query}
        Account information:
        Plan: {plan}
        Current users: {current_user_count}
        Available slots: {available_slots}
        Please provide a helpful response to this account management question.
        """
        return self.llm_utils.generate_response(prompt, self.system_prompt)

class AgentOrchestrator:
    def __init__(
        self,
        llm_utils: LLMUtils,
        knowledge_base: Dict[str, Any],
        vector_db: Dict[str, str],
    ):
        self.llm_utils = llm_utils
        self.knowledge_base = knowledge_base
        self.vector_db = vector_db
        self.conversations = {}
        self.router_agent = RouterAgent(llm_utils)
        self.product_agent = ProductSpecialistAgent(
            llm_utils,
            knowledge_base["product_catalog"],
            knowledge_base["faqs"],
            vector_db["products"],
        )
        self.technical_agent = TechnicalSupportAgent(
            llm_utils, knowledge_base["tech_docs"], vector_db["technical"]
        )
        self.billing_agent = OrderBillingAgent(
            llm_utils, knowledge_base["product_catalog"]
        )
        self.account_agent = AccountManagementAgent(llm_utils)
        logger.info("Agent Orchestrator initialized with all agents")

    async def process_query(
        self, query: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        conversation_history = self.conversations.get(conversation_id, [])
        routing_result = self.router_agent.process(query, conversation_history)
        if routing_result.get("multi_part", False):
            responses = []
            for part in routing_result.get("parts", []):
                part_query = part.get("query_part")
                part_classification = part.get("classification")
                part_response = await self._process_single_query(
                    part_query, part_classification, conversation_history
                )
                responses.append(f"{part_response}")
            final_response = "\n\n".join(responses)
            agent_type = "multiple"
        else:
            classification = routing_result.get("classification", "General")
            if routing_result.get("requires_clarification", False):
                final_response = routing_result.get(
                    "clarification_question",
                    "Could you please provide more details about your question?",
                )
                agent_type = "router"
            else:
                final_response = await self._process_single_query(
                    query, classification, conversation_history
                )
                agent_type = classification.lower()
        self.conversations[conversation_id].append(
            {"query": query, "response": final_response, "agent": agent_type}
        )
        return {
            "response": final_response,
            "agent": agent_type,
            "conversation_id": conversation_id or "new_conversation",
        }

    async def _process_single_query(
        self,
        query: str,
        classification: str,
        conversation_history: List[Dict[str, Any]],
    ) -> str:
        if classification == "Product":
            return await self.product_agent.process(query, conversation_history)
        elif classification == "Technical":
            return await self.technical_agent.process(query, conversation_history)
        elif classification == "Billing":
            return await self.billing_agent.process(query, conversation_history)
        elif classification == "Account":
            return await self.account_agent.process(query, conversation_history)
        else:
            general_prompt = f"""
            Customer query: {query}
            Please provide a helpful and friendly general response to this query.
            """
            general_system_prompt = """
            You are a Customer Support Agent for TechSolutions.
            Provide helpful, friendly, and concise responses to general customer inquiries.
            If the query should be handled by a specialist agent, indicate which type of specialist would be appropriate.
            """
            return self.llm_utils.generate_response(
                general_prompt, general_system_prompt
            )