import hashlib
import os
import json
import datetime
from pathlib import Path

class IPManifestGenerator:
    """
    Secures LUMINARK Intellectual Property.
    Generates a cryptographic manifest of the entire codebase for timestamping.
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.ignore_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.env', '.vscode'}
        self.critical_files = []
        
    def generate_manifest(self) -> Dict[str, Any]:
        print("🔒 Scanning LUMINARK codebase for IP protection...")
        
        manifest = {
            "project": "LUMINARK ANTIKYTHERA",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "generated_by": "Antigravity Agent",
            "files": {},
            "total_hash": ""
        }
        
        all_hashes = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Filtering
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for file in files:
                if file.endswith(('.py', '.md', '.html', '.css', '.js', '.json')):
                    filepath = Path(root) / file
                    try:
                        file_hash = self._hash_file(filepath)
                        rel_path = str(filepath.relative_to(self.root_dir))
                        
                        manifest["files"][rel_path] = file_hash
                        all_hashes.append(file_hash)
                        print(f"   - Secured: {rel_path}")
                    except Exception as e:
                        print(f"   ! Error securing {file}: {e}")
                        
        # Create a Master Hash of all file hashes
        # This single hash proves the existence of the entire codebase state
        master_hash = hashlib.sha256(("".join(sorted(all_hashes))).encode('utf-8')).hexdigest()
        manifest["total_hash"] = master_hash
        
        print(f"\n🔐 MASTER HASH GENERATED: {master_hash}")
        return manifest
        
    def _hash_file(self, filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def save_manifest(self, manifest: Dict[str, Any], output_path: str):
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"✅ IP Manifest saved to: {output_path}")
        print("   -> Upload this JSON or the Master Hash to OpenTimestamps.org to prove ownership.")

if __name__ == "__main__":
    # Run from project root
    root = os.getcwd()
    generator = IPManifestGenerator(root)
    manifest = generator.generate_manifest()
    
    # Save to security folder
    output = Path(root) / "luminark" / "security" / "ip_manifest_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    generator.save_manifest(manifest, str(output))
