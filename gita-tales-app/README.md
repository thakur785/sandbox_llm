# 🎯 Gita Tales: AI-Powered Animated Bhagavad Gita for Kids

An innovative AI application that converts Sanskrit shlokas from the Bhagavad Gita into engaging, animated stories for children. This project is part of the **Hybrid LLM Learning Roadmap (2025-2027)**.

## 🌟 Project Vision

Transform ancient Sanskrit wisdom into modern, interactive learning experiences for children using cutting-edge AI technologies including:
- **Text Processing**: Convert shlokas into kid-friendly stories
- **Multi-Modal AI**: Text → Story → Visuals → Animation → Audio
- **Voice Generation**: Multi-language narration with emotion
- **Character Creation**: AI-generated animated characters (Krishna, Arjuna, etc.)
- **Interactive Learning**: Q&A with AI Krishna, moral lesson games

## 🚀 Phase 1 Features (Aug - Dec 2025)

### ✅ Current Implementation
- [x] Basic shloka-to-story conversion using OpenAI GPT
- [x] Development environment setup
- [x] Progress tracking system
- [x] Multi-language support foundation

### 🔄 In Progress
- [ ] Streamlit web interface
- [ ] ElevenLabs text-to-speech integration
- [ ] Google Translate multi-language support
- [ ] Character description generator

### ⏳ Upcoming
- [ ] Animation pipeline planning
- [ ] Image generation for characters
- [ ] Interactive Q&A system
- [ ] Mobile app development

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Web), React Native (Mobile - Phase 2)
- **Backend**: Python, FastAPI
- **AI Models**: OpenAI GPT-4, ElevenLabs Voice AI
- **Translation**: Google Translate API
- **Animation**: Planned integration with video generation AI
- **Database**: SQLite (local), PostgreSQL (production)

## 📁 Project Structure

```
gita-tales-app/
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── gita_story_generator.py   # Core story generation logic
├── learning_tracker.py      # Progress tracking system
├── README.md                # This file
├── data/                    # Shloka database (coming soon)
├── streamlit_app.py         # Web interface (coming soon)
├── audio/                   # Generated voice files
├── images/                  # Character images
└── tests/                   # Unit tests
```

## 🎯 Learning Goals Alignment

This project serves as a practical implementation for mastering:

### **Month 1: Foundation (Aug - Dec 2025)**
- **Week 1-2**: Python + LLM APIs, Basic text processing
- **Week 3-4**: Multi-language support, Voice generation
- **Week 5-8**: Web interface, Character creation
- **Week 9-12**: Animation planning, Interactive features

### **Learning Split**: 40% theory, 60% hands-on building
- **Weekly Target**: 6.4 hours learning + 9.6 hours building = 16 hours total

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and navigate to project
cd gita-tales-app

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies (already done in your setup)
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# ELEVENLABS_API_KEY=your_key_here
```

### 3. Test Basic Functionality
```bash
# Run the story generator
python gita_story_generator.py

# Track your learning progress
python learning_tracker.py
```

## 📊 Progress Tracking

Use the built-in learning tracker to monitor your progress:

```python
from learning_tracker import LearningTracker

tracker = LearningTracker()
tracker.log_progress(
    week=1, 
    learning_hours=3.0, 
    building_hours=5.0, 
    notes="Completed basic setup and first story generation",
    tasks_completed=["Set up environment", "Create story generator"]
)
```

## 🎨 Sample Output

**Input Shloka:**
```sanskrit
धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।
मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय।।
```

**Generated Story (for 8-year-old):**
*Title: The Great Question at Kurukshetra*

Once upon a time, there was a wise king who was worried about a big family problem. Two groups of cousins couldn't agree on how to share their kingdom fairly...

## 🎯 Next Steps (Week 1-2)

1. **Today**: Set up OpenAI API key and test story generation
2. **This Weekend**: Explore ElevenLabs for voice generation
3. **Next Week**: Build Streamlit interface for easy story creation
4. **End of August**: Have working prototype with voice narration

## 📚 Learning Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [ElevenLabs Python SDK](https://elevenlabs.io/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Bhagavad Gita Online](https://www.holy-bhagavad-gita.org/)

## 🤝 Contributing

This is a personal learning project following the Hybrid LLM Learning Roadmap. Feel free to:
- Suggest improvements
- Share ideas for new features
- Provide feedback on story quality
- Contribute translations

## 📄 License

MIT License - Feel free to use this code for your own learning journey!

## 🎉 Acknowledgments

- Inspired by the timeless wisdom of the Bhagavad Gita
- Part of the Hybrid LLM Learning Roadmap (2025-2027)
- Built with love for making ancient wisdom accessible to modern children

---

**Happy Learning! 🚀 Let's make this journey epic!**