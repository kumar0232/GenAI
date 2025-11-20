import json
import logging
import os
from typing import Any, Dict, List

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_dir: str = "data", supabase_url: str = "", supabase_key: str = ""):
        self.data_dir = data_dir
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase = None
        self.embedder = None

        # Initialize embedder if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SentenceTransformer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize SentenceTransformer: {e}")

        # Initialize Supabase client if credentials provided
        if supabase_url and supabase_key and SUPABASE_AVAILABLE:
            try:
                self.supabase = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase (will use fallback): {e}")
        else:
            logger.info("Supabase not configured, using fallback storage")

    @staticmethod
    def split_text(text, chunk_size=1000, chunk_overlap=200, separators=None):
        """Custom text splitter without LangChain."""
        if not separators:
            separators = ["\n## ", "\n### ", "\n#### ", "\n", ". ", "! ", "? ", ";", ":", " ", ""]
        chunks = []
        text_length = len(text)
        i = 0

        while i < text_length:
            min_split_pos = text_length
            selected_separator = ""
            for sep in separators:
                next_pos = text.find(sep, i, i + chunk_size)
                if next_pos != -1 and next_pos < min_split_pos:
                    min_split_pos = next_pos
                    selected_separator = sep

            if min_split_pos == text_length:
                min_split_pos = min(i + chunk_size, text_length)

            current_chunk = text[i:min_split_pos + len(selected_separator)]
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            i = min_split_pos - chunk_overlap if min_split_pos - chunk_overlap > i else min_split_pos

            if i >= text_length:
                break

        return chunks

    def load_knowledge_base(self) -> Dict[str, Any]:
        """Load and prepare knowledge base for agents"""
        try:
            # Load product catalog
            product_catalog_path = os.path.join(self.data_dir, "product_catalog.json")
            if os.path.exists(product_catalog_path):
                with open(product_catalog_path, "r") as f:
                    product_catalog = json.load(f)
            else:
                logger.warning("Product catalog not found, using empty catalog")
                product_catalog = {"products": [], "addons": [], "bundles": []}

            # Load FAQs
            faq_path = os.path.join(self.data_dir, "faq.json")
            if os.path.exists(faq_path):
                with open(faq_path, "r") as f:
                    faqs = json.load(f)
            else:
                logger.warning("FAQ not found, using empty FAQ")
                faqs = {"categories": []}

            # Load tech docs
            tech_docs_path = os.path.join(self.data_dir, "tech_documentation.md")
            if os.path.exists(tech_docs_path):
                with open(tech_docs_path, "r") as f:
                    tech_docs = f.read()
            else:
                logger.warning("Tech documentation not found, using empty docs")
                tech_docs = "# Technical Documentation\n\nNo documentation available."

            # Load customer conversations
            conversations_path = os.path.join(self.data_dir, "customer_conversations.jsonl")
            customer_conversations = []
            if os.path.exists(conversations_path):
                with open(conversations_path, "r") as f:
                    for line in f:
                        if line.strip():
                            try:
                                customer_conversations.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse conversation line: {e}")
            else:
                logger.warning("Customer conversations not found, using empty list")

            logger.info("Knowledge base loaded successfully")
            return {
                "product_catalog": product_catalog,
                "faqs": faqs,
                "tech_docs": tech_docs,
                "customer_conversations": customer_conversations,
            }
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            # Return minimal fallback structure
            return {
                "product_catalog": {"products": [], "addons": [], "bundles": []},
                "faqs": {"categories": []},
                "tech_docs": "# Technical Documentation\n\nDocumentation not available.",
                "customer_conversations": [],
            }

    def prepare_vector_db(self, knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare vector database collections for each type of data"""
        if not self.supabase:
            logger.info("Using fallback vector storage (in-memory)")
            return self._prepare_fallback_vector_db(knowledge_base)
        
        collections = {}
        try:
            collections["products"] = self._prepare_product_collection(
                knowledge_base["product_catalog"], knowledge_base["faqs"]
            )
            collections["technical"] = self._prepare_technical_collection(
                knowledge_base["tech_docs"]
            )
            collections["conversations"] = self._prepare_conversations_collection(
                knowledge_base["customer_conversations"]
            )
        except Exception as e:
            logger.error(f"Failed to prepare Supabase collections, using fallback: {e}")
            return self._prepare_fallback_vector_db(knowledge_base)
        
        return collections

    def _prepare_fallback_vector_db(self, knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback in-memory vector database"""
        return {
            "products": "products_fallback",
            "technical": "technical_fallback", 
            "conversations": "conversations_fallback",
            "fallback_data": {
                "products": knowledge_base["product_catalog"],
                "faqs": knowledge_base["faqs"],
                "tech_docs": knowledge_base["tech_docs"],
                "conversations": knowledge_base["customer_conversations"]
            }
        }

    def _prepare_conversations_collection(self, conversations: List[Dict[str, Any]]) -> str:
        """Prepare vector collection for customer conversations"""
        if not self.supabase or not self.embedder:
            return "conversations_fallback"
        
        try:
            collection_name = "conversations"

            # Check if collection already has data
            try:
                response = self.supabase.table(collection_name).select("id").limit(1).execute()
                if response.data:
                    logger.info("Conversations collection already populated, skipping")
                    return collection_name
            except Exception:
                pass

            docs, metas, ids = [], [], []

            for i, convo in enumerate(conversations):
                convo_text = f"Customer: {convo.get('customer')}\nAgent: {convo.get('agent')}\nConversation: {convo.get('text', '')}"
                chunks = self.split_text(convo_text)
                for j, chunk in enumerate(chunks):
                    docs.append(chunk)
                    metas.append({
                        "type": "conversation",
                        "conversation_id": convo.get("id", ""),
                        "chunk": f"{i}-{j}",
                    })
                    ids.append(f"convo-{convo.get('id', '')}-{j}")

            # Batch insert for speed
            batch_size = 10
            for start in range(0, len(docs), batch_size):
                batch = []
                for doc, meta, doc_id in zip(
                    docs[start:start+batch_size], metas[start:start+batch_size], ids[start:start+batch_size]
                ):
                    try:
                        embedding = self.embedder.encode(doc).tolist()
                        batch.append({
                            "id": doc_id,
                            "content": doc,
                            "metadata": meta,
                            "embedding": embedding
                        })
                    except Exception as e:
                        logger.warning(f"Failed to encode document {doc_id}: {e}")

                if batch:
                    self.supabase.table(collection_name).insert(batch).execute()

            logger.info(f"Added {len(docs)} conversation documents to Supabase")
            return collection_name

        except Exception as e:
            logger.error(f"Error preparing conversations collection: {e}")
            return "conversations_fallback"


    def _prepare_technical_collection(self, tech_docs: str) -> str:
        """Prepare vector collection for technical documentation"""
        if not self.supabase or not self.embedder:
            return "technical_fallback"
            
        try:
            collection_name = "technical"
            
            # Check if collection already exists
            try:
                response = self.supabase.table(collection_name).select("id").limit(1).execute()
                if response.data:
                    logger.info("Technical collection already populated, skipping")
                    return collection_name
            except Exception:
                pass

            chunks = self.split_text(tech_docs)
            
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.embedder.encode(chunk).tolist()
                    self.supabase.table(collection_name).insert({
                        "id": f"tech-{i}",
                        "content": chunk,
                        "metadata": {"type": "technical_doc", "chunk": str(i)},
                        "embedding": embedding
                    }).execute()
                except Exception as e:
                    logger.warning(f"Failed to insert tech doc chunk {i}: {e}")

            logger.info(f"Added {len(chunks)} technical documents to Supabase")
            return collection_name
            
        except Exception as e:
            logger.error(f"Error preparing technical collection: {e}")
            return "technical_fallback"

  