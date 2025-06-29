#!/usr/bin/env python3
import argparse
from typing import Dict, Any
import logging
from rag_engine import RAGEngine
from document_parser import DocumentProcessor, get_processing_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('civil_code_rag.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_system():
    return True

def initialize_vector_stores(force_recreate: bool = False):
    """Initialize vector stores for all countries"""
    print("\n📚 Initializing document processing...")
    
    doc_processor = DocumentProcessor()
    
    # Check current status
    status = get_processing_status()
    
    if not force_recreate and all(status.values()):
        print("✅ All vector stores already exist")
        return doc_processor
    
    print("🔄 Creating vector stores (this may take a while)...")
    
    try:
        doc_processor.initialize_all_vector_stores(force_recreate)
        print("✅ Vector stores initialized successfully")
        return doc_processor
    except Exception as e:
        print(f"❌ Error initializing vector stores: {e}")
        return None

def interactive_query():
    """Interactive query interface"""
    print("\n💬 Interactive Query Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 40)
    
    rag_engine = RAGEngine()
    
    while True:
        try:
            query = input("\n🤔 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print_help()
                continue
            
            if not query:
                continue
            
            print("🔍 Processing your query with Gemini 2.0 Flash...")
            
            # Process query
            result = rag_engine.query_with_country_detection(query)
            
            # Display results
            display_player_result(result)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.error(f"Query processing error: {e}", exc_info=True)

def display_query_results(result: Dict[str, Any]):
    """Display query results in a formatted way"""
    print("\n" + "=" * 60)
    print("📋 QUERY RESULTS")
    print("=" * 60)
    
    # Country detection info
    if "country_detection" in result:
        detection = result["country_detection"]
        print(f"🎯 Country Detection:")
        print(f"   Predicted: {detection['predicted_country']}")
        print(f"   Confidence: {detection['confidence']:.2f}")
        print(f"   Strategy: {detection['search_strategy']}")
        
        if detection['search_strategy'] == 'multiple_countries':
            print(f"   Countries searched: {', '.join(detection['searched_countries'])}")
    
    # Results
    if "results_by_country" in result:
        # Multiple countries
        for country, country_result in result["results_by_country"].items():
            print(f"\n🏛️  {country.upper()} RESULTS:")
            display_single_country_result(country_result)
    else:
        # Single country
        display_single_country_result(result)

def display_single_country_result(result: Dict[str, Any]):
    """Display results for a single country"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Translation info
    if result.get("translated_query") and result.get("translated_query") != result.get("original_query"):
        print(f"🌐 Query translated: {result['original_query']} -> {result['translated_query']}")
    
    # Gemini response
    if "gemini_response" in result:
        response = result["gemini_response"]
        print(f"\n🤖 Gemini 2.0 Flash Response:")
        
        if response.get("error"):
            print(f"❌ Error: {response['error']}")
        else:
            print(f"⏱️  Processing time: {response['processing_time']:.2f}s")
            
            # Show translated response if different from original
            if response.get("translated_response") != response.get("original_response"):
                print(f"📝 Translated Response:\n{response['translated_response']}")
                print(f"\n📝 Original Response:\n{response['original_response']}")
            else:
                print(f"📝 Response:\n{response['translated_response']}")
    
    # Context info
    if result.get("source_info"):
        print(f"\n📚 Sources used: {len(result['source_info'])} document chunks")

def print_help():
    """Print help information"""
    print("""
Available commands:
- Just type your legal question in any supported language
- 'quit' or 'exit' - Exit the program
- 'help' - Show this help

Supported languages:
- French (France)
- German (Germany)  
- Italian (Italy)
- Spanish (Spain)
- Portuguese (Portugal)
- English (queries will be translated to appropriate language)

Example queries:
- "What is property law?"
- "Qu'est-ce que le droit de propriété?"
- "Was ist Eigentumsrecht?"
- "Qual è il diritto di proprietà?"
- "¿Cuál es el derecho de propiedad?"

Note: This simplified version uses only Gemini 2.0 Flash and free translation services.
""")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Player Performance Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                 # Initial setup
  python main.py --init-vectors          # Initialize vector stores
  python main.py --test                  # Run all tests
  python main.py --query "What is property law?"  # Single query
  python main.py --interactive           # Interactive mode
        """
    )
    
    parser.add_argument('--setup', action='store_true', 
                       help='Setup and validate system configuration')
    parser.add_argument('--init-vectors', action='store_true',
                       help='Initialize vector stores for all countries')
    parser.add_argument('--force-recreate', action='store_true',
                       help='Force recreate vector stores even if they exist')
    parser.add_argument('--test', action='store_true',
                       help='Run system tests')
    parser.add_argument('--query', type=str,
                       help='Single query to process')
    parser.add_argument('--country', type=str,
                       help='Specific country to query (use with --query)')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive query mode')
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle different modes
    if args.setup:
        if not setup_system():
            sys.exit(1)
        return
    
    if args.init_vectors:
        if not setup_system():
            sys.exit(1)
        doc_processor = initialize_vector_stores(args.force_recreate)
        if not doc_processor:
            sys.exit(1)
        return
    
    if args.test:
        if not setup_system():
            sys.exit(1)
        test_translation()
        test_country_classification()
        return
    
    
    
    
    
    if args.query:
        if not setup_system():
            sys.exit(1)
        
        # Initialize RAG engine
        rag_engine = RAGEngine()
        
        # Process query
        
            result = rag_engine.query_single_country(args.query, args.country)
        else:
            result = rag_engine.query_player_data(args.query)
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                # Convert complex objects to JSON-serializable format
                serializable_result = convert_to_serializable(result)
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to {args.output}")
        else:
            display_player_result(result)
        return
    
    if args.interactive:
        if not setup_system():
            sys.exit(1)
        interactive_query()
        return
    
    # Default: show help
    parser.print_help()

def convert_to_serializable(obj):
    """Convert complex objects to JSON-serializable format"""
    if hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dictionary
        result = {}
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue  # Skip private attributes
            if callable(value):
                continue  # Skip methods
            result[key] = convert_to_serializable(value)
        return result
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

if __name__ == "__main__":
    main()

def display_player_result(result: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("📋 QUERY RESULTS")
    print("=" * 60)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        if "user_message" in result:
            print(f"💬 Message: {result['user_message']}")
        return

    print(f"⏱️  Processing time: {result['gemini_response']['processing_time']:.2f}s")
    print(f"🧠 Gemini Response:\n{result['gemini_response']['text']}")
    print(f"📊 Rows used for context: {result['context_rows']}")
