"""
Mock Story Generator - Test your app without API costs
"""

from datetime import datetime
import json

def mock_story_generator():
    """Test the story generator without using OpenAI API"""
    
    sample_shloka = """धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।
मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय।।1.1।।"""
    
    mock_story = """
🌟 **The Great Question at Kurukshetra** 🌟
*A Story for 8-year-olds*

**Setting:** Long, long ago, there was a big, green field called Kurukshetra. It was like a huge playground, but today it looked different - many people had gathered there.

**Story:**
Once upon a time, there was a blind king named Dhritarashtra who lived in a beautiful palace. He had 100 sons, but he was worried because his sons and their cousins (called the Pandavas) couldn't agree on how to share their family's kingdom.

The king called his trusted friend Sanjaya and asked, "What are my sons and the Pandava brothers doing in that big field? I'm worried about what might happen."

You see, sometimes in families, people disagree about things. The king's sons thought they should rule the whole kingdom, but the Pandavas also had a right to their share. Instead of talking nicely to solve the problem, both groups went to the field to settle it their own way.

Krishna, who was very wise and kind, was there to help his friend Arjuna (one of the Pandavas) understand what was right and wrong.

**Moral Lesson:**
When families have disagreements, it's always better to talk and find peaceful solutions rather than fighting. Sometimes we need wise friends to help us understand what is the right thing to do.

**Discussion Questions:**
1. What would you do if you had a disagreement with your brother or sister?
2. Who do you talk to when you need help making good decisions?
3. Why is it important to be fair when sharing with others?

**Animation Ideas:**
- Scene 1: A beautiful palace with the worried king
- Scene 2: The green field with many people gathering
- Scene 3: Krishna and Arjuna talking as friends
- Scene 4: A happy ending showing families talking peacefully
    """
    
    # Create story data structure (same as real generator)
    story_data = {
        "timestamp": datetime.now().isoformat(),
        "shloka_original": sample_shloka,
        "shloka_meaning": "In the field of dharma, in Kurukshetra, what did my sons and Pandavas do?",
        "child_age": 8,
        "language": "English",
        "generated_story": mock_story,
        "model_used": "mock-storyteller",
        "tokens_used": 0,
        "cost": 0.0
    }
    
    return story_data

def save_mock_story(story_data):
    """Save the mock story to see the output format"""
    filename = f"mock_story_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(story_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Mock story saved to: {filename}")
    return filename

if __name__ == "__main__":
    print("🎭 Testing with Mock Story Generator...")
    print("=" * 60)
    
    # Generate mock story
    story = mock_story_generator()
    
    print("📖 Generated Story:")
    print("-" * 40)
    print(story["generated_story"])
    
    # Save the story
    save_mock_story(story)
    
    print("\n" + "=" * 60)
    print("✅ Mock test complete!")
    print("💡 Once you add billing to OpenAI, replace 'mock' with real API calls")
    print("🎯 Your app structure is working perfectly!")