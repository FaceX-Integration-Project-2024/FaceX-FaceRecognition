from supabase import create_client

def create_supabase_client(url, key):
    return create_client(url, key)
