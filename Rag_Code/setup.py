"""Setup script to install requirements and initialize database"""
import subprocess
import sys
import os

def install_requirements():
    """Install all requirements"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def initialize_database():
    """Initialize database and seed data"""
    print("\nInitializing database...")
    try:
        from src.database.db_models import init_db
        from src.utils.seed_data import seed_all
        
        init_db()
        print("✓ Database initialized")
        
        print("Seeding data...")
        seed_all()
        print("✓ Data seeded successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_env_file():
    """Check if .env file exists and has API key"""
    if not os.path.exists('.env'):
        print("⚠ .env file not found. Please create it and add your GEMINI_API_KEY")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY', '')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("⚠ Please update .env file with your GEMINI_API_KEY")
        return False
    
    print("✓ .env file configured")
    return True

if __name__ == '__main__':
    print("=" * 50)
    print("Hybrid RAG AI Agent - Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("\n✗ Setup failed at requirements installation")
        sys.exit(1)
    
    # Check environment
    check_env_file()
    
    # Initialize database
    if not initialize_database():
        print("\n✗ Setup failed at database initialization")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Make sure Neo4j is running")
    print("2. Update .env file with your GEMINI_API_KEY and Neo4j password")
    print("3. Run: python app.py")
    print("=" * 50)

