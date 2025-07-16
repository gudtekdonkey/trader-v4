#!/usr/bin/env python3
"""
Database initialization script
Creates all necessary tables and indexes
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://trader:trader_password@localhost:5432/trading')


def create_database():
    """Create the trading database if it doesn't exist"""
    # Parse database URL
    parts = DATABASE_URL.split('/')
    db_name = parts[-1]
    base_url = '/'.join(parts[:-1])
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(f"{base_url}/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute(f"CREATE DATABASE {db_name}")
        print(f"Created database: {db_name}")
    else:
        print(f"Database {db_name} already exists")
    
    cursor.close()
    conn.close()


def initialize_tables():
    """Create all necessary tables"""
    engine = create_engine(DATABASE_URL)
    
    # Read SQL schema file
    schema_file = os.path.join(os.path.dirname(__file__), '..', 'docker', 'init-db.sql')
    
    if os.path.exists(schema_file):
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema
        with engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
        
        print("Database schema initialized successfully")
    else:
        print(f"Schema file not found: {schema_file}")
        print("Creating basic schema...")
        
        # Basic schema if file not found
        basic_schema = """
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(50) NOT NULL,
            side VARCHAR(10) NOT NULL,
            price DECIMAL(20, 8) NOT NULL,
            quantity DECIMAL(20, 8) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS positions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(50) NOT NULL,
            quantity DECIMAL(20, 8) NOT NULL,
            entry_price DECIMAL(20, 8) NOT NULL,
            status VARCHAR(20) DEFAULT 'open',
            opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS market_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            open DECIMAL(20, 8) NOT NULL,
            high DECIMAL(20, 8) NOT NULL,
            low DECIMAL(20, 8) NOT NULL,
            close DECIMAL(20, 8) NOT NULL,
            volume DECIMAL(20, 8) NOT NULL,
            timeframe VARCHAR(10) NOT NULL
        );
        """
        
        with engine.connect() as conn:
            conn.execute(text(basic_schema))
            conn.commit()
        
        print("Basic schema created")


def verify_setup():
    """Verify database setup"""
    engine = create_engine(DATABASE_URL)
    
    try:
        with engine.connect() as conn:
            # Check tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            
            tables = [row[0] for row in result]
            
            print("\nDatabase tables:")
            for table in tables:
                print(f"  - {table}")
            
            # Check if tables have data
            for table in ['trades', 'positions', 'market_data']:
                if table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    print(f"  - {table}: {count} records")
        
        print("\nDatabase setup verified successfully!")
        
    except Exception as e:
        print(f"Error verifying database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Initializing database...")
    
    try:
        # Create database
        create_database()
        
        # Initialize tables
        initialize_tables()
        
        # Verify setup
        verify_setup()
        
        print("\nDatabase initialization completed!")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        sys.exit(1)