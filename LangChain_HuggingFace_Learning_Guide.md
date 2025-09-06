# Gita Tales AI - LangChain & HuggingFace Learning Journey
# Week 2-3: Advanced LLM Application Development

## 🎯 Learning Objectives
- Master LangChain for building LLM applications
- Explore HuggingFace ecosystem for model management
- Integrate both into your Gita Tales project
- Build production-ready AI pipelines

## 📚 Week 2: LangChain Fundamentals

### Day 1: Basic LangChain Setup
```python
# Install required libraries
!pip install langchain openai transformers datasets accelerate

# Basic imports
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Set up OpenAI
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# Initialize LLM
llm = OpenAI(temperature=0.7)
```

### Day 2: Prompt Engineering for Gita Stories
```python
# Create specialized prompts for your project
gita_story_template = PromptTemplate(
    input_variables=["shloka", "age", "language", "moral_focus"],
    template="""
    Convert this Bhagavad Gita shloka into an engaging story for a {age}-year-old child:
    
    Shloka: {shloka}
    Language: {language}
    Moral Focus: {moral_focus}
    
    Create a story with:
    1. Simple, age-appropriate language
    2. Krishna and Arjuna as relatable characters
    3. Clear moral lesson about {moral_focus}
    4. Engaging dialogue and scenes
    5. Happy, encouraging ending
    
    Story:
    """
)

# Create chain
story_chain = LLMChain(llm=llm, prompt=gita_story_template)

# Test with sample shloka
sample_shloka = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन"
story = story_chain.run(
    shloka=sample_shloka,
    age=8,
    language="English",
    moral_focus="doing your best without worrying about results"
)
print(story)
```

### Day 3: Document Processing for Gita Text
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load and process Gita text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Create vector store for semantic search
embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_texts(gita_chunks, embeddings)

# Semantic search for relevant shlokas
def find_relevant_shlokas(query, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    return docs
```

### Day 4: Memory and Conversation
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Add memory for interactive storytelling
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Interactive Q&A about the story
def interactive_gita_chat():
    print("🕉️ Ask Krishna anything about the story!")
    while True:
        question = input("You: ")
        if question.lower() in ['quit', 'exit', 'bye']:
            break
        
        response = conversation.predict(input=f"""
        You are Krishna from the Bhagavad Gita, speaking to a child. 
        Answer this question in a simple, wise, and caring way:
        
        Question: {question}
        """)
        print(f"Krishna: {response}")
```

## 📚 Week 3: HuggingFace Integration

### Day 1: Model Exploration
```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Explore different models for text generation
models_to_try = [
    "gpt2",
    "microsoft/DialoGPT-medium",
    "ai4bharat/indic-bart-hi-en",  # For Hindi translation
    "google/flan-t5-base"
]

# Text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Test story generation
prompt = "Once upon a time, Krishna taught Arjuna about"
generated = generator(prompt, max_length=100, num_return_sequences=2)
for story in generated:
    print(story['generated_text'])
```

### Day 2: Multi-language Support
```python
# Translation pipeline for multi-language stories
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

def translate_story(english_story):
    # Translate to Hindi
    hindi_story = translator(english_story, max_length=1000)
    return hindi_story[0]['translation_text']

# Test translation
english_story = "Krishna smiled and said to Arjuna, 'Do your duty without attachment.'"
hindi_story = translate_story(english_story)
print(f"English: {english_story}")
print(f"Hindi: {hindi_story}")
```

### Day 3: Custom Model Fine-tuning
```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset

# Prepare dataset for fine-tuning on Gita stories
def prepare_gita_dataset():
    # Sample data structure
    stories = [
        {"shloka": "कर्मण्येवाधिकारस्ते", "story": "Krishna teaches about action..."},
        # Add more story pairs
    ]
    
    dataset = Dataset.from_list(stories)
    return dataset

# Fine-tuning setup (simplified)
def setup_fine_tuning():
    training_args = TrainingArguments(
        output_dir="./gita-storyteller",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10
    )
    return training_args
```

## 🎯 Project Integration Tasks

### Task 1: Upgrade Your Story Generator
```python
# Enhanced story generator using LangChain
class AdvancedGitaStoryGenerator:
    def __init__(self):
        self.llm = OpenAI(temperature=0.7)
        self.story_chain = LLMChain(llm=self.llm, prompt=gita_story_template)
        self.memory = ConversationBufferMemory()
    
    def generate_story(self, shloka, age=8, language="English"):
        story = self.story_chain.run(
            shloka=shloka,
            age=age,
            language=language,
            moral_focus="life lessons"
        )
        return story
    
    def interactive_qa(self, story_context):
        # Add interactive Q&A about the generated story
        pass
```

### Task 2: Multi-modal Pipeline
```python
# Complete pipeline: Text -> Story -> Translation -> Audio
class GitaTalesMultiModalPipeline:
    def __init__(self):
        self.story_generator = AdvancedGitaStoryGenerator()
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
        # Add TTS pipeline later
    
    def create_complete_story(self, shloka, age=8, target_language="Hindi"):
        # Generate English story
        english_story = self.story_generator.generate_story(shloka, age)
        
        # Translate if needed
        if target_language != "English":
            translated_story = self.translator(english_story)
            final_story = translated_story[0]['translation_text']
        else:
            final_story = english_story
        
        return {
            "original_shloka": shloka,
            "english_story": english_story,
            "final_story": final_story,
            "language": target_language,
            "age_group": age
        }
```

## 🎯 Weekly Goals Checklist

### Week 2 - LangChain Mastery:
- [ ] Set up LangChain in Colab
- [ ] Create advanced prompt templates
- [ ] Implement document processing
- [ ] Add conversation memory
- [ ] Build interactive Q&A

### Week 3 - HuggingFace Integration:
- [ ] Explore pre-trained models
- [ ] Implement multi-language translation
- [ ] Set up model fine-tuning pipeline
- [ ] Create custom model for Gita stories
- [ ] Optimize for production deployment

### Integration Tasks:
- [ ] Upgrade existing story generator
- [ ] Create multi-modal pipeline
- [ ] Add semantic search for shlokas
- [ ] Implement story quality evaluation
- [ ] Build comprehensive testing suite

## 📊 Learning Resources Summary

**LangChain Deep Dive:**
- Official Documentation: https://python.langchain.com/docs/
- GitHub Examples: https://github.com/langchain-ai/langchain/tree/master/cookbook
- DeepLearning.AI Course: https://www.deeplearning.ai/short-courses/

**HuggingFace Mastery:**
- HuggingFace Course: https://huggingface.co/course/
- Model Hub: https://huggingface.co/models
- Transformers Docs: https://huggingface.co/docs/transformers/

**Colab-Specific Resources:**
- Colab Tips & Tricks: https://colab.research.google.com/notebooks/basic_features_overview.ipynb
- GPU Usage Guide: https://research.google.com/colaboratory/faq.html

## 🚀 Next Steps After Week 3:
1. Deploy models to production
2. Create Streamlit interface with advanced features
3. Add image generation for story illustrations
4. Implement voice narration with emotion
5. Build mobile app prototype

Happy Learning! 🎯🚀