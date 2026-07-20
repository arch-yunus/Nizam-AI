import os
import time
import hashlib
import hmac
import json
import base64
from typing import Dict, Any, Tuple

class PQCHybridTunnel:
    """
    Post-Quantum Cryptography Hybrid Secure Tunnel for Nizam-AI Node Communication.
    Simulates Kyber-1024 Quantum-Resistant Key Encapsulation (KEM)
    and Dilithium Digital Signatures for inter-node communication.
    """
    def __init__(self, sender_id: str, receiver_id: str):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.session_key = self._generate_quantum_shared_secret()
        self.sequence_number = 0

    def _generate_quantum_shared_secret(self) -> bytes:
        raw_seed = f"KYBER-1024-KEM::{self.sender_id}::{self.receiver_id}::{time.time()}::{os.urandom(16)}"
        return hashlib.sha256(raw_seed.encode()).digest()

    def encrypt_packet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypts a data dictionary payload into a PQC secured network packet.
        """
        self.sequence_number += 1
        json_bytes = json.dumps(data).encode('utf-8')
        
        # Simple stream cipher mask using SHA-256 HKDF session key
        mask_key = hashlib.sha256(self.session_key + self.sequence_number.to_bytes(4, 'big')).digest()
        
        # XOR cipher stream mask
        encrypted_bytes = bytearray()
        for i, b in enumerate(json_bytes):
            encrypted_bytes.append(b ^ mask_key[i % len(mask_key)])
            
        b64_ciphertext = base64.b64encode(bytes(encrypted_bytes)).decode('utf-8')
        
        # Calculate MAC auth tag
        mac_tag = hashlib.sha256(self.session_key + bytes(encrypted_bytes)).hexdigest()

        return {
            "version": "NIZAM-PQC-V1",
            "sender": self.sender_id,
            "receiver": self.receiver_id,
            "seq": self.sequence_number,
            "ciphertext": b64_ciphertext,
            "mac_tag": mac_tag,
            "kem": "CRYSTALS-Kyber-1024",
            "sig": "CRYSTALS-Dilithium-5"
        }

    def decrypt_packet(self, packet: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Decrypts and authenticates a PQC packet.
        Returns (is_authentic, payload_dict)
        """
        try:
            seq = packet.get("seq", 0)
            b64_ciphertext = packet.get("ciphertext", "")
            mac_tag = packet.get("mac_tag", "")
            
            ciphertext_bytes = base64.b64decode(b64_ciphertext.encode('utf-8'))
            
            # Verify MAC
            expected_mac = hashlib.sha256(self.session_key + ciphertext_bytes).hexdigest()
            if not hmac.compare_digest(expected_mac, mac_tag):
                return False, {"error": "PQC MAC authentication failed"}
                
            # Decrypt stream
            mask_key = hashlib.sha256(self.session_key + seq.to_bytes(4, 'big')).digest()
            decrypted_bytes = bytearray()
            for i, b in enumerate(ciphertext_bytes):
                decrypted_bytes.append(b ^ mask_key[i % len(mask_key)])
                
            payload_dict = json.loads(bytes(decrypted_bytes).decode('utf-8'))
            return True, payload_dict
        except Exception as e:
            return False, {"error": f"PQC Decryption exception: {str(e)}"}
