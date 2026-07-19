import random
from typing import Dict, List, Tuple
from nizam.hybrid import HybridAIModel
from nizam.integrations import T3AIClient

class PersonalizedLearningAssistant:
    """
    Simulates a localized, personalized AI learning assistant.
    Adapts educational difficulty and questions dynamically based on student focus.
    """
    def __init__(self):
        # Features: [attention_level (0-100), math_score (0-100), science_score (0-100), fatigue_level (0-100)]
        # Classes: ['ReviewMode' (Beginner), 'ReinforceMode' (Intermediate), 'ChallengeMode' (Advanced)]
        # Rules:
        # If fatigue_level (feature 3) > 75.0 => ReviewMode (class 0)
        # If math_score (feature 1) > 85.0 AND science_score (feature 2) > 85.0 AND fatigue_level < 40.0 => ChallengeMode (class 2)
        rules = {
            0: [(3, 1.0, 75.0)],  # Fatigue > 75 -> Review
            2: [(1, 1.0, 85.0), (2, 1.0, 85.0), (3, 0.0, 40.0)] # High grades, low fatigue -> Challenge
        }
        self.model = HybridAIModel(
            num_features=4,
            classes=['ReviewMode', 'ReinforceMode', 'ChallengeMode'],
            rules=rules
        )
        self.t3ai_client = T3AIClient()
        
        # Local questions database
        self.questions = {
            'ReviewMode': [
                "2x + 5 = 15 ise x nedir? Adım adım çözelim.",
                "Suyun donma sıcaklığı kaç derecedir?",
                "DNA'nın temel yapı birimleri nelerdir?"
            ],
            'ReinforceMode': [
                "x^2 - 4x + 4 = 0 denkleminin kökleri nelerdir?",
                "Yerçekimi ivmesi Dünya'da neden kutuplara yakın bölgelerde daha büyüktür?",
                "Fotosentez reaksiyonlarının temel basamaklarını açıklayınız."
            ],
            'ChallengeMode': [
                "Euler sayısının (e) türevsel tanımlamasını yaparak lim x->inf (1 + 1/x)^x limitini gösteriniz.",
                "Genel görelilik teorisindeki uzay-zaman bükülmesinin Newtonyen yerçekimi kuramından farklarını tartışınız.",
                "Kuantum dolanıklık ilkesinin yerel gerçekçilik (local realism) paradoksuna getirdiği açıklamaları Küre Ansiklopedisi kaynaklarıyla tartışınız."
            ]
        }

    def assess_student_state(self, attention: float, math_score: float, science_score: float, fatigue: float) -> Dict:
        """
        Classifies state of student and provides tailored questions.
        """
        features = [attention, math_score, science_score, fatigue]
        recommended_mode, confidence, telemetry = self.model.predict(features, statistical_weight=0.6, force_symbolic=True)
        
        # Pick a question matching the recommended mode
        questions_pool = self.questions[recommended_mode]
        base_question = random.choice(questions_pool)
        
        # Enrich the question using simulated T3AI
        prompt = f"Gerekli Mod: {recommended_mode}. Konu: '{base_question}'. Lütfen bu soruyu Türkçe dil yapısına ve seviyeye uygun şekilde detaylandır."
        t3ai_res = self.t3ai_client.generate(prompt)
        enriched_question = t3ai_res["choices"][0]["text"]

        return {
            "student_metrics": {
                "attention_level": attention,
                "math_score": math_score,
                "science_score": science_score,
                "fatigue_level": fatigue
            },
            "recommended_mode": recommended_mode,
            "confidence": round(confidence, 2),
            "original_question": base_question,
            "t3ai_enriched_prompt": enriched_question,
            "telemetry": telemetry
        }
