#!/usr/bin/env python3
"""
Redis Setup Script for Triage.Flow
Helps install and configure Redis for optimal AI/RAG caching
"""
import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command with error handling"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e.stderr}")
        return None

def detect_system():
    """Detect the operating system"""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return "unknown"

def install_redis():
    """Install Redis based on the operating system"""
    system = detect_system()
    
    print(f"🖥️  Detected system: {system}")
    
    if system == "macos":
        # macOS with Homebrew
        print("📦 Installing Redis via Homebrew...")
        run_command("brew install redis", "Installing Redis")
        run_command("brew services start redis", "Starting Redis service")
        
    elif system == "linux":
        # Ubuntu/Debian
        print("📦 Installing Redis via apt...")
        run_command("sudo apt update", "Updating package lists")
        run_command("sudo apt install -y redis-server", "Installing Redis")
        run_command("sudo systemctl start redis-server", "Starting Redis service")
        run_command("sudo systemctl enable redis-server", "Enabling Redis on boot")
        
    elif system == "windows":
        print("📦 For Windows, please install Redis manually:")
        print("1. Download Redis from: https://github.com/microsoftarchive/redis/releases")
        print("2. Or use WSL2 with Ubuntu and run this script again")
        print("3. Or use Docker: docker run -d -p 6379:6379 redis:alpine")
        
    else:
        print("❓ Unknown system. Please install Redis manually.")

def configure_redis():
    """Configure Redis for optimal AI/RAG performance"""
    print("⚙️  Configuring Redis for AI workloads...")
    
    config_optimizations = """
# Redis configuration optimizations for AI/RAG workloads
maxmemory 2gb
maxmemory-policy allkeys-lru
save 300 10
appendonly yes
appendfsync everysec

# Increase client timeout for large operations
timeout 300

# Optimize for memory efficiency
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
"""
    
    # Try to find Redis config file locations
    config_paths = [
        "/usr/local/etc/redis.conf",  # macOS Homebrew
        "/etc/redis/redis.conf",      # Linux
        "/etc/redis.conf",            # Alternative Linux
    ]
    
    config_found = False
    for config_path in config_paths:
        if os.path.exists(config_path):
            print(f"📝 Found Redis config at: {config_path}")
            print("⚠️  Please manually add these optimizations to your Redis config:")
            print(config_optimizations)
            config_found = True
            break
    
    if not config_found:
        print("📝 Redis config file not found in standard locations.")
        print("💡 Create a custom redis.conf with these optimizations:")
        print(config_optimizations)

def test_redis_connection():
    """Test Redis connection"""
    print("🧪 Testing Redis connection...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful!")
        
        # Test some operations
        r.set('test_key', 'test_value', ex=60)
        value = r.get('test_key')
        r.delete('test_key')
        
        if value == b'test_value':
            print("✅ Redis read/write operations working!")
        else:
            print("❌ Redis read/write test failed")
            
    except ImportError:
        print("⚠️  redis-py not installed. Installing...")
        run_command(f"{sys.executable} -m pip install redis", "Installing redis-py")
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Make sure Redis is running: redis-server")

def create_env_template():
    """Create .env template with Redis configuration"""
    env_template = """
# Redis Configuration for Triage.Flow
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
# REDIS_PASSWORD=your_password_if_needed
# REDIS_SSL=false

# Cache Configuration
CACHE_ENABLED=true
ENABLE_RAG_CACHING=true
ENABLE_RESPONSE_CACHING=true
CACHE_TTL_RAG=1800
CACHE_TTL_RESPONSE=3600
CACHE_TTL_FOLDER=1800

# Performance Settings
MAX_CACHE_SIZE=2000
MAX_CACHE_MEMORY_MB=1000
"""
    
    env_file = Path(".env.redis.example")
    with open(env_file, "w") as f:
        f.write(env_template.strip())
    
    print(f"📄 Created {env_file} with Redis configuration template")
    print("💡 Copy relevant settings to your .env file")

def main():
    """Main setup function"""
    print("🚀 Redis Setup for Triage.Flow AI Caching")
    print("=" * 50)
    
    # Check if Redis is already installed
    redis_installed = run_command("redis-cli --version", "Checking Redis installation")
    
    if not redis_installed:
        print("📦 Redis not found. Installing...")
        install_redis()
    else:
        print("✅ Redis already installed")
    
    # Configure Redis
    configure_redis()
    
    # Test connection
    test_redis_connection()
    
    # Create environment template
    create_env_template()
    
    print("\n🎉 Redis setup complete!")
    print("\n📋 Next steps:")
    print("1. Copy Redis settings from .env.redis.example to your .env file")
    print("2. Restart your Triage.Flow application")
    print("3. Monitor cache performance at /cache-stats endpoint")
    print("4. For production, consider Redis Cluster or Redis Sentinel")

if __name__ == "__main__":
    main() 