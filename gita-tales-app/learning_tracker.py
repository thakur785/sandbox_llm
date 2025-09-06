from datetime import datetime, timedelta
import json
import os

class LearningTracker:
    """Track progress according to the Hybrid LLM Learning Roadmap"""
    
    def __init__(self):
        self.start_date = datetime(2025, 8, 23)  # Your actual start date
        self.data_file = "learning_progress.json"
        self.weekly_goals = {
            "week_1": {
                "learning_hours": 6.4,
                "building_hours": 9.6,
                "milestone": "Basic Gita text processor with OpenAI API",
                "tasks": [
                    "Set up development environment",
                    "Create basic story generator",
                    "Test with sample shloka",
                    "Implement error handling"
                ],
                "completed": False
            },
            "week_2": {
                "learning_hours": 6.4,
                "building_hours": 9.6, 
                "milestone": "Multi-language support and TTS integration",
                "tasks": [
                    "Add Google Translate integration",
                    "Implement ElevenLabs voice generation",
                    "Create multilingual story function",
                    "Test voice narration"
                ],
                "completed": False
            },
            "week_3": {
                "learning_hours": 6.4,
                "building_hours": 9.6,
                "milestone": "Streamlit web interface",
                "tasks": [
                    "Create Streamlit app interface",
                    "Add shloka input functionality",
                    "Implement age selector",
                    "Add story display and audio playback"
                ],
                "completed": False
            },
            "week_4": {
                "learning_hours": 6.4,
                "building_hours": 9.6,
                "milestone": "Character and animation planning",
                "tasks": [
                    "Design character descriptions",
                    "Create storyboard templates",
                    "Plan animation pipeline",
                    "Research image generation tools"
                ],
                "completed": False
            }
        }
        self.load_progress()
    
    def load_progress(self):
        """Load existing progress from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    saved_data = json.load(f)
                    # Update weekly goals with saved progress
                    for week, data in saved_data.get('weekly_goals', {}).items():
                        if week in self.weekly_goals:
                            self.weekly_goals[week].update(data)
            except Exception as e:
                print(f"Could not load progress: {e}")
    
    def save_progress(self):
        """Save progress to JSON file"""
        data = {
            'start_date': self.start_date.isoformat(),
            'weekly_goals': self.weekly_goals,
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            print("✅ Progress saved successfully!")
        except Exception as e:
            print(f"❌ Could not save progress: {e}")
    
    def get_current_week(self):
        """Calculate which week we're currently in"""
        days_passed = (datetime.now() - self.start_date).days
        current_week = (days_passed // 7) + 1
        return max(1, min(current_week, 4))  # Clamp between 1-4 for Month 1
    
    def log_progress(self, week, learning_hours, building_hours, notes="", tasks_completed=None):
        """Log progress for a specific week"""
        current_week = f"week_{week}"
        if current_week in self.weekly_goals:
            # Update hours
            self.weekly_goals[current_week]['actual_learning_hours'] = learning_hours
            self.weekly_goals[current_week]['actual_building_hours'] = building_hours
            self.weekly_goals[current_week]['notes'] = notes
            self.weekly_goals[current_week]['date_logged'] = datetime.now().isoformat()
            
            # Update completed tasks
            if tasks_completed:
                self.weekly_goals[current_week]['completed_tasks'] = tasks_completed
            
            # Check if week is completed
            total_target = self.weekly_goals[current_week]['learning_hours'] + self.weekly_goals[current_week]['building_hours']
            total_actual = learning_hours + building_hours
            self.weekly_goals[current_week]['completed'] = total_actual >= total_target * 0.8  # 80% completion
            
            self.save_progress()
            self.display_week_summary(week)
        else:
            print(f"❌ Week {week} not found in goals")
    
    def display_week_summary(self, week):
        """Display progress summary for a week"""
        current_week = f"week_{week}"
        if current_week in self.weekly_goals:
            data = self.weekly_goals[current_week]
            print(f"\n📊 Week {week} Progress Summary")
            print("=" * 40)
            print(f"🎯 Milestone: {data['milestone']}")
            print(f"📚 Learning: {data.get('actual_learning_hours', 0):.1f}/{data['learning_hours']} hours")
            print(f"🔨 Building: {data.get('actual_building_hours', 0):.1f}/{data['building_hours']} hours")
            print(f"✅ Status: {'COMPLETED' if data.get('completed', False) else 'IN PROGRESS'}")
            
            if 'notes' in data and data['notes']:
                print(f"📝 Notes: {data['notes']}")
            
            print(f"\n📋 Tasks:")
            for i, task in enumerate(data['tasks'], 1):
                completed_tasks = data.get('completed_tasks', [])
                status = "✅" if task in completed_tasks else "⏳"
                print(f"   {status} {task}")
            
            print("=" * 40)
    
    def display_overall_progress(self):
        """Display overall progress for Month 1"""
        print(f"\n🚀 Gita Tales AI - Learning Progress Tracker")
        print(f"📅 Started: {self.start_date.strftime('%B %d, %Y')}")
        print(f"📅 Current Date: {datetime.now().strftime('%B %d, %Y')}")
        print(f"📅 Current Week: {self.get_current_week()}")
        print("=" * 50)
        
        total_learning = 0
        total_building = 0
        completed_weeks = 0
        
        for week_num in range(1, 5):
            week_key = f"week_{week_num}"
            data = self.weekly_goals[week_key]
            
            actual_learning = data.get('actual_learning_hours', 0)
            actual_building = data.get('actual_building_hours', 0)
            total_learning += actual_learning
            total_building += actual_building
            
            if data.get('completed', False):
                completed_weeks += 1
            
            status = "✅ DONE" if data.get('completed', False) else "🔄 IN PROGRESS" if self.get_current_week() >= week_num else "⏳ UPCOMING"
            print(f"Week {week_num}: {status} - {data['milestone']}")
        
        print("=" * 50)
        print(f"📊 Overall Statistics:")
        print(f"   📚 Total Learning Hours: {total_learning:.1f} / 25.6")
        print(f"   🔨 Total Building Hours: {total_building:.1f} / 38.4")
        print(f"   ✅ Completed Weeks: {completed_weeks} / 4")
        print(f"   📈 Progress: {(completed_weeks / 4) * 100:.1f}%")

def main():
    """Example usage of the Learning Tracker"""
    tracker = LearningTracker()
    
    print("Welcome to your LLM Learning Journey! 🚀")
    tracker.display_overall_progress()
    
    # Get current week
    current_week = tracker.get_current_week()
    print(f"\nYou are currently in Week {current_week}")
    
    # Log today's progress (August 24, 2025)
    if current_week == 1:
        print("\n🎉 Logging your progress for today...")
        tracker.log_progress(
            week=1, 
            learning_hours=2.5, 
            building_hours=4.0, 
            notes="Set up secure development environment, learned API security best practices, created story generator with error handling, implemented mock testing",
            tasks_completed=[
                "Set up development environment", 
                "Create basic story generator",
                "Set up secure API key management (.env)",
                "Create comprehensive project structure",
                "Implement error handling and testing"
            ]
        )
        
        print("\n💡 What you've learned today:")
        print("  📚 Environment variables and security")
        print("  📚 Python classes and error handling") 
        print("  📚 OpenAI API integration")
        print("  📚 Git branching and professional workflow")
        print("  📚 JSON data structures and file I/O")
        
        print("\n🎯 Next session goals:")
        print("  🔨 Add billing to OpenAI account")
        print("  🔨 Test real story generation")
        print("  🔨 Start Streamlit web interface")
        print("  🔨 Implement age-specific story variations")
    else:
        print(f"\n📝 Ready to log progress for Week {current_week}")
        print("Uncomment the log_progress lines below and update with your actual hours and tasks")

def quick_progress_check():
    """Quick function to check today's progress without logging"""
    tracker = LearningTracker()
    current_week = tracker.get_current_week()
    
    print(f"📅 Current Week: {current_week}")
    print(f"📊 Target for Week {current_week}: 6.4 learning + 9.6 building = 16 hours")
    
    if tracker.weekly_goals.get(f"week_{current_week}"):
        week_data = tracker.weekly_goals[f"week_{current_week}"]
        
        actual_learning = week_data.get('actual_learning_hours', 0)
        actual_building = week_data.get('actual_building_hours', 0)
        target_learning = week_data['learning_hours'] 
        target_building = week_data['building_hours']
        
        print(f"📚 Learning Progress: {actual_learning}/{target_learning} hours ({(actual_learning/target_learning)*100:.1f}%)")
        print(f"🔨 Building Progress: {actual_building}/{target_building} hours ({(actual_building/target_building)*100:.1f}%)")
        
        remaining_learning = max(0, target_learning - actual_learning)
        remaining_building = max(0, target_building - actual_building)
        
        if remaining_learning > 0 or remaining_building > 0:
            print(f"\n⏰ Remaining this week:")
            if remaining_learning > 0:
                print(f"  📚 Learning: {remaining_learning:.1f} hours")
            if remaining_building > 0:
                print(f"  🔨 Building: {remaining_building:.1f} hours")
        else:
            print("🎉 Week 1 targets achieved!")

def log_additional_hours(learning_hours=0, building_hours=0, notes=""):
    """Quick function to add more hours to current week"""
    tracker = LearningTracker()
    current_week = tracker.get_current_week()
    
    week_key = f"week_{current_week}"
    if week_key in tracker.weekly_goals:
        current_learning = tracker.weekly_goals[week_key].get('actual_learning_hours', 0)
        current_building = tracker.weekly_goals[week_key].get('actual_building_hours', 0)
        
        new_learning = current_learning + learning_hours
        new_building = current_building + building_hours
        
        tracker.log_progress(
            week=current_week,
            learning_hours=new_learning,
            building_hours=new_building,
            notes=notes,
            tasks_completed=tracker.weekly_goals[week_key].get('completed_tasks', [])
        )

if __name__ == "__main__":
    print("🚀 LLM Learning Journey Tracker")
    print("Choose an option:")
    print("1. Show full progress summary")
    print("2. Quick progress check")
    print("3. Log additional hours")
    
    try:
        choice = input("\nEnter choice (1-3) or press Enter for option 1: ").strip()
        
        if choice == "2":
            quick_progress_check()
        elif choice == "3":
            print("\nLogging additional hours...")
            learning = float(input("Additional learning hours: ") or "0")
            building = float(input("Additional building hours: ") or "0") 
            notes = input("Notes (optional): ").strip()
            log_additional_hours(learning, building, notes)
        else:
            main()  # Default option
            
    except KeyboardInterrupt:
        print("\n👋 See you next time!")
    except ValueError:
        print("❌ Please enter valid numbers for hours")