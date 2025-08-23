"""
Gita Tales: AI Storyteller - Core Story Generation Module
Convert Sanskrit shlokas from Bhagavad Gita into engaging stories for kids
"""

import openai
import os
from dotenv import load_dotenv
from datetime import datetime
import json

# Load environment variables
load_dotenv()

class GitaStoryGenerator:
    def __init__(self):
        """Initialize the Gita Story Generator with OpenAI API"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("⚠️  Warning: OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        # Set up OpenAI client
        openai.api_key = self.api_key

    def generate_kid_story(self, shloka_text, shloka_meaning="", child_age=8, language="English"):
        """
        Convert Sanskrit shloka from Bhagavad Gita into a simple, engaging story
        for a child of specified age.
        
        Args:
            shloka_text (str): The Sanskrit shloka
            shloka_meaning (str): Optional translation/meaning
            child_age (int): Age of the target child (5-12)
            language (str): Target language for the story
            
        Returns:
            dict: Generated story with metadata
        """
        
        # Age-appropriate vocabulary and concepts
        age_guidelines = {
            5: "Very simple words, 2-3 sentence stories, focus on colors and actions",
            6: "Simple sentences, basic emotions, clear good vs bad concepts",
            7: "Short paragraphs, friendship themes, simple moral lessons",
            8: "Engaging narratives, character development, clear life lessons",
            9: "Detailed stories, complex emotions, multiple characters",
            10: "Rich narratives, deeper meanings, cultural context",
            11: "Advanced concepts, philosophical discussions, historical context",
            12: "Complex themes, detailed explanations, preparation for original text"
        }
        
        age_guide = age_guidelines.get(child_age, age_guidelines[8])
        
        prompt = f"""
        Convert this Sanskrit shloka from Bhagavad Gita into an engaging story for a {child_age}-year-old child in {language}.
        
        Sanskrit Shloka: {shloka_text}
        
        {f'Meaning: {shloka_meaning}' if shloka_meaning else ''}
        
        Age Guidelines: {age_guide}
        
        Create a story that includes:
        1. **Setting**: Describe the scene (battlefield, palace, forest, etc.) in vivid, child-friendly terms
        2. **Characters**: Krishna and Arjuna as main characters with clear personalities
        3. **Dialogue**: Simple, engaging conversations between characters
        4. **Action**: What happens in the story that illustrates the shloka's teaching
        5. **Moral Lesson**: Clear, age-appropriate life lesson that children can apply
        6. **Animation Cues**: Brief descriptions of visual scenes for future animation
        
        Format the response as a structured story with:
        - Title
        - Setting description
        - Story narrative (3-5 short paragraphs)
        - Moral lesson
        - Discussion questions for parents/teachers
        
        Keep vocabulary appropriate for a {child_age}-year-old and make it culturally respectful.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert storyteller specializing in adapting ancient wisdom texts for children. You create engaging, culturally respectful, and age-appropriate stories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            generated_story = response.choices[0].message.content
            
            # Create story metadata
            story_data = {
                "timestamp": datetime.now().isoformat(),
                "shloka_original": shloka_text,
                "shloka_meaning": shloka_meaning,
                "child_age": child_age,
                "language": language,
                "generated_story": generated_story,
                "model_used": "gpt-4",
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
            return story_data
            
        except Exception as e:
            error_message = f"Error generating story: {str(e)}"
            print(f"❌ {error_message}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": error_message,
                "shloka_original": shloka_text,
                "child_age": child_age,
                "language": language
            }
    
    def save_story(self, story_data, filename=None):
        """Save generated story to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gita_story_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Story saved to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving story: {str(e)}")
            return None

# Test function to demonstrate the generator
def test_story_generation():
    """Test the story generator with a sample shloka"""
    
    # Sample shloka from Bhagavad Gita (Chapter 1, Verse 1)
    test_shloka = """धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।
मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय।।1.1।।"""
    
    test_meaning = """In the field of dharma, in Kurukshetra, gathered together and eager to fight, 
what did my sons and the sons of Pandu do, O Sanjaya?"""
    
    print("🚀 Testing Gita Story Generator...")
    print("=" * 60)
    
    generator = GitaStoryGenerator()
    
    # Generate story for an 8-year-old
    result = generator.generate_kid_story(
        shloka_text=test_shloka,
        shloka_meaning=test_meaning,
        child_age=8,
        language="English"
    )
    
    if "error" not in result:
        print("✅ Story Generated Successfully!")
        print("\n📖 Generated Story:")
        print("-" * 40)
        print(result["generated_story"])
        
        # Save the story
        generator.save_story(result)
    else:
        print(f"❌ Error: {result['error']}")
        print("\n💡 Make sure to:")
        print("1. Create a .env file with your OPENAI_API_KEY")
        print("2. Get an OpenAI API key from https://platform.openai.com/api-keys")

if __name__ == "__main__":
    test_story_generation()