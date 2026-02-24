#!/usr/bin/env python3
"""
Initialize the EEG ADHD Detection System database.
Run this once to set up all tables and initial data.
"""

import mysql.connector
from mysql.connector import Error

# Database connection config (root user to create database)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Update if you have a root password
}

# Read the SQL schema
with open('database_schema.sql', 'r') as f:
    schema_sql = f.read()

try:
    # Connect to MySQL
    conn = mysql.connector.connect(**DB_CONFIG, auth_plugin='mysql_native_password')
    cursor = conn.cursor()
    
    print("Connected to MySQL server...")
    print("Creating database and tables...")
    
    # Execute all SQL statements
    for statement in schema_sql.split(';'):
        statement = statement.strip()
        if statement:
            try:
                cursor.execute(statement)
                print(f"✓ Executed: {statement[:80]}...")
            except Error as err:
                if "already exists" in str(err) or "Duplicate" in str(err):
                    print(f"⚠ Skipped (already exists): {statement[:50]}...")
                else:
                    print(f"✗ Error: {err}")
    
    # Commit changes
    conn.commit()
    print("\n✓ Database initialization complete!")
    
except Error as e:
    print(f"Connection Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure MySQL is running")
    print("2. Check that root user has no password (or update the password in DB_CONFIG)")
    print("3. On Windows, MySQL might need to be started via Services")
finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
