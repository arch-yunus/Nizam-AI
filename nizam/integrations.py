import time
import random
from typing import Dict, Any

class T3AIClient:
    """
    Simulated Client for T3AI (Türkiye's Sovereign Big Language Model).
    Generates culturally aligned, secure Turkish language outputs.
    """
    def __init__(self, api_url: str = "https://api.t3ai.org.tr/v1"):
        self.api_url = api_url

    def generate(self, prompt: str, system_instruction: str = None) -> Dict[str, Any]:
        """
        Simulates generation from T3AI.
        """
        # Simulated delay
        time.sleep(0.1)
        
        # Simple responses based on keywords in prompt
        prompt_lower = prompt.lower()
        
        if "nizam" in prompt_lower or "nedir" in prompt_lower:
            response = ("Nizam-AI, açık kaynaklı, milli teknoloji hamlesi vizyonuyla şekillendirilmiş "
                        "ve yerel cihazlarda çalışabilen melez bir yapay zeka mimarisidir.")
        elif "savunma" in prompt_lower or "uav" in prompt_lower or "iha" in prompt_lower:
            response = ("T3AI Savunma Katmanı: Otonom sürülerin yerel taktik değerlendirmeleri, "
                        "karıştırıcı unsurlara karşı elektronik harp dayanıklılığı sağlayacak şekilde "
                        "Nizam-AI algoritmalarıyla entegre edilmiştir.")
        elif "sağlık" in prompt_lower or "kanser" in prompt_lower:
            response = ("T3AI Medikal Çözümler: Hücresel analizler ve ilaç adayları yerel uç donanımlarda "
                        "milli mahremiyet standartlarına uygun olarak analiz edilmektedir.")
        else:
            response = (f"T3AI Cevabı: '{prompt}' talebiniz milli egemenlik ve güvenli "
                        f"veri sınırları kapsamında yerel modelimiz tarafından başarıyla işlendi.")

        return {
            "model": "t3ai-sovereign-v1",
            "choices": [{
                "text": response,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            },
            "cultural_alignment": "100% Verified Turkish Language & Cultural Heritage Context"
        }


class EnSosyalClient:
    """
    Simulated Client for En Sosyal.
    A safe, community-oriented social network for publishing non-toxic, factual node updates.
    """
    def __init__(self, api_url: str = "https://api.ensosyal.org.tr"):
        self.api_url = api_url
        self.posts_history = []

    def publish_post(self, title: str, content: str, author_node: str) -> Dict[str, Any]:
        """
        Simulates publishing an event to En Sosyal.
        """
        post = {
            "post_id": f"post_{random.randint(10000, 99999)}",
            "author": author_node,
            "title": title,
            "content": content,
            "timestamp": time.time(),
            "toxicity_score": 0.00,  # En Sosyal is non-toxic by design
            "status": "Published"
        }
        self.posts_history.append(post)
        return post


class KureClient:
    """
    Simulated Client for Küre (The open-source, scientific digital encyclopedia).
    Provides verified facts to prevent AI hallucinations.
    """
    def __init__(self, api_url: str = "https://kure.org.tr/api"):
        self.api_url = api_url
        # Local verified facts database
        self.knowledge_base = {
            "uav_swarm_flocking": "Sürü İHA'lar birbirleriyle yerel RF bandı üzerinden mesafe ve yön bilgilerini "
                                  "paylaşarak çarpışmayı önler ve toplu hedeflere yönelir.",
            "quantization_loss": "Quantization (nicemleme), float32 ağırlıkların int8 formatına indirilmesiyle %75 "
                                 "bellek tasarrufu sağlar, doğruluk kaybı genellikle %1-2 civarındadır.",
            "cellular_diagnosis": "Akciğer nodül sınıflandırmalarında yerel derin ağlar, hücre sınırlarını segmentize "
                                  "ederek hekime hızlı ön tanı raporu sunar."
        }

    def query_fact(self, keyword: str) -> Dict[str, Any]:
        """
        Queries verified facts.
        """
        for k, v in self.knowledge_base.items():
            if keyword.lower() in k.lower():
                return {
                    "found": True,
                    "keyword": k,
                    "fact": v,
                    "source": "Küre Ansiklopedisi (Akademik Kurul Onaylı)",
                    "last_updated": "2026-05-12"
                }
        
        return {
            "found": False,
            "keyword": keyword,
            "fact": "Arama kelimesine ait doğrulanmış bir akademik kayıt bulunamadı. Yapay zeka uydurmasını engellemek için varsayılan güvenli moda geçildi.",
            "source": "Küre - Bilgi Tabanı Güvenliği"
        }
