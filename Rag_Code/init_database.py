"""Initialize database and seed data"""
import os
import sys
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Initialize database and seed data"""
    try:
        logger.info("Initializing database...")
        from src.database.db_models import init_db
        init_db()
        logger.info("✓ Database initialized")
        
        logger.info("Seeding ChromaDB and Neo4j...")
        from src.utils.seed_data import seed_all
        success = seed_all()
        
        if success:
            logger.info("✓ Database initialization completed successfully!")
        else:
            logger.warning("⚠ Database initialization completed with some warnings")
            
    except Exception as e:
        logger.error(f"✗ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

