import os
import subprocess
import sqlite3

def vulnerable_function(user_input):
    # SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    
    # Command injection vulnerability
    os.system(f"echo {user_input}")
    
    # Potential path traversal
    filename = f"/tmp/{user_input}.txt"
    with open(filename, 'w') as f:
        f.write("data")
    
    return result

def quality_issues():
    # High complexity function
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i == j == k:
                    print("complex logic")
    
    # Poor variable naming
    x = 1
    y = 2
    z = x + y
    return z

# Potential AI hallucination - fictional library
import nonexistent_lib
result = nonexistent_lib.magic_function()