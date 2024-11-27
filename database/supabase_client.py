from supabase import create_client, Client

def create_supabase_client(url, key) -> Client:
    return create_client(url, key)
