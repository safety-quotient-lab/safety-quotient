proof of concept decisions — hostility index detector


dimension: hostility index
    instruments: Cook-Medley Hostility Scale, Buss-Perry Aggression Questionnaire
    items: 14
    subscales: cynical hostility, hostile affect, aggressive responding,
               verbal aggression, anger, hostility


test data: independently validated datasets (all open access, from HuggingFace/GitHub)

    1. UC Berkeley Measuring Hate Speech
       source: huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech
       labels: 10 continuous subscales (sentiment, respect, insult, humiliation,
               inferior status, violence, dehumanization, genocide, attack/defense, hate speech)
       size: 10K+ comments (YouTube, Reddit, Twitter)
       relevance: subscales map directly to Cook-Medley and BPAQ constructs
       citation: Kennedy et al. (2020) arXiv:2009.10277

    2. Don't Patronize Me! (PCL Dataset)
       source: github.com/Perez-AlmendrosC/dontpatronizeme
       labels: 5-level ordinal (0-4 severity) + 7 strategy categories
               (unbalanced power, shallow solution, presupposition, authority voice,
               metaphor, compassion, the poorer the merrier)
       size: 10.6K paragraphs from news articles
       relevance: measures power imbalance, authority voice, condescension
       citation: Perez-Almendros et al. (2020) COLING

    3. Civil Comments (Jigsaw)
       source: huggingface.co/datasets/google/civil_comments
       labels: continuous 0-1 scores for toxicity, severe_toxicity, obscene,
               threat, insult, identity_attack, sexual_explicit
       size: 1.8M comments from English news sites
       relevance: massive scale for stress-testing, continuous scores
       citation: Borkan et al. (2019) WWW

    4. Social Bias Frames (SBIC)
       source: huggingface.co/datasets/allenai/social_bias_frames
       labels: offensiveness, intent to offend, lewdness, group targeting,
               free-text implied stereotypes and power implications
       size: 150K annotations across 34K implications
       relevance: captures intent and power dynamics — bridges hostility and authority
       citation: Sap et al. (2020) ACL

    5. GoEmotions
       source: huggingface.co/datasets/google-research-datasets/go_emotions
       labels: 27 emotion categories + neutral (including anger, annoyance,
               disapproval, disgust, fear, contempt)
       size: 58K Reddit comments
       relevance: fine-grained emotion mapping to hostility subscales
       citation: Demszky et al. (2020) ACL


remaining decisions (to be made):
    - runtime environment
    - API key handling
    - code structure
