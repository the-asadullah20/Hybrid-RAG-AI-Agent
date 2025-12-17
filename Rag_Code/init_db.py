"""Initialize database and seed data"""
import logging
from src.database.db_models import init_db
from src.utils.seed_data import seed_all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")
    
    logger.info("Seeding data...")
    seed_all()
    logger.info("Initialization complete!")

