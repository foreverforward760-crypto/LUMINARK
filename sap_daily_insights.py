"""
SAP DAILY INSIGHTS - Email Newsletter System
Sends daily SAP wisdom + stage insights to subscribers

Setup:
1. Use Mailchimp, ConvertKit, or SendGrid
2. Run this script daily (cron job or scheduled task)
3. Builds audience for upselling to paid products
"""

import random
from datetime import datetime
from typing import List, Dict

class SAPDailyInsights:
    """Generates daily SAP content for email newsletter"""
    
    def __init__(self):
        self.stage_wisdom = {
            0: {
                "quote": "In the void, all possibilities exist.",
                "insight": "Plenara teaches us that emptiness isn't absence—it's potential.",
                "practice": "Today, sit with uncertainty. Don't rush to fill the void."
            },
            1: {
                "quote": "Every journey begins with a single spark.",
                "insight": "The Spark stage reminds us that small beginnings matter.",
                "practice": "Notice what's emerging. What new possibility is calling you?"
            },
            2: {
                "quote": "Duality creates clarity, but also limitation.",
                "insight": "Polarity helps us see choices, but most decisions aren't binary.",
                "practice": "Where are you seeing either/or? Can you find both/and?"
            },
            3: {
                "quote": "Motion creates momentum, but direction matters.",
                "insight": "Action is powerful, but purposeless motion wastes energy.",
                "practice": "Check in: Are you moving toward something or away from something?"
            },
            4: {
                "quote": "A strong foundation supports growth.",
                "insight": "Foundation provides stability, but can become a cage.",
                "practice": "What structures serve you? Which ones limit you?"
            },
            5: {
                "quote": "At the threshold, everything changes.",
                "insight": "Stage 5 is the point of no return. Choose wisely.",
                "practice": "What decision are you avoiding? What would happen if you chose?"
            },
            6: {
                "quote": "Integration transcends duality.",
                "insight": "Both/and thinking is more sophisticated than either/or.",
                "practice": "Find the paradox. Hold two truths simultaneously."
            },
            7: {
                "quote": "Question everything, including your questions.",
                "insight": "Healthy skepticism prevents dogma.",
                "practice": "What are you certain about? Challenge one certainty today."
            },
            8: {
                "quote": "Certainty is the beginning of stagnation.",
                "insight": "Stage 8 trap: When success becomes rigidity.",
                "practice": "⚠️ Where are you too certain? Invite doubt."
            },
            9: {
                "quote": "Let go to make space for what's next.",
                "insight": "Renewal requires release. Death precedes rebirth.",
                "practice": "What are you holding onto? What wants to be released?"
            }
        }
        
        self.current_events_templates = [
            "Markets showing Stage {stage} patterns: {description}",
            "Political climate reflects Stage {stage}: {description}",
            "Cultural moment: Stage {stage} dynamics at play: {description}",
            "Tech industry displaying Stage {stage} characteristics: {description}"
        ]
    
    def generate_daily_email(self) -> Dict[str, str]:
        """Generate today's email content"""
        
        # Rotate through stages (or pick based on current events)
        day_of_year = datetime.now().timetuple().tm_yday
        stage = day_of_year % 10  # Cycles through 0-9
        
        wisdom = self.stage_wisdom[stage]
        
        # Generate email
        subject = f"SAP Daily Insight: {wisdom['quote']}"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; text-align: center;">
                <h1 style="margin: 0;">🌟 SAP Daily Insight</h1>
                <p style="font-size: 1.2rem; margin: 10px 0;">{datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            
            <div style="padding: 30px; background: #f8f9fa; border-radius: 15px; margin: 20px 0;">
                <h2 style="color: #667eea;">Stage {stage}: Today's Focus</h2>
                <blockquote style="font-size: 1.3rem; font-style: italic; color: #333; border-left: 4px solid #667eea; padding-left: 20px;">
                    "{wisdom['quote']}"
                </blockquote>
                
                <h3 style="color: #764ba2;">💡 Insight</h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">{wisdom['insight']}</p>
                
                <h3 style="color: #764ba2;">🎯 Today's Practice</h3>
                <p style="font-size: 1.1rem; line-height: 1.6; background: white; padding: 15px; border-radius: 10px;">
                    {wisdom['practice']}
                </p>
            </div>
            
            <div style="background: #667eea; color: white; padding: 20px; border-radius: 15px; text-align: center;">
                <h3>Want Personalized SAP Guidance?</h3>
                <p>Take our free assessment to discover your current stage</p>
                <a href="https://sapframework.com/assessment" style="display: inline-block; background: white; color: #667eea; padding: 12px 30px; border-radius: 25px; text-decoration: none; font-weight: bold; margin: 10px 0;">
                    Take Free Assessment →
                </a>
            </div>
            
            <div style="text-align: center; padding: 20px; color: #666; font-size: 0.9rem;">
                <p>SAP Framework | Stanfield's Axiom of Perpetuity</p>
                <p><a href="https://sapframework.com/unsubscribe" style="color: #666;">Unsubscribe</a></p>
            </div>
        </body>
        </html>
        """
        
        return {
            'subject': subject,
            'body': body,
            'stage': stage
        }
    
    def generate_week_ahead(self) -> str:
        """Generate weekly preview email"""
        
        subject = "SAP Weekly Preview: Your 7-Day Journey"
        
        # Get next 7 stages
        day_of_year = datetime.now().timetuple().tm_yday
        stages = [(day_of_year + i) % 10 for i in range(7)]
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        preview_html = ""
        for day, stage in zip(days, stages):
            wisdom = self.stage_wisdom[stage]
            preview_html += f"""
            <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 10px;">
                <strong>{day} - Stage {stage}:</strong> {wisdom['quote']}
            </div>
            """
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <h1 style="color: #667eea;">🌟 Your SAP Week Ahead</h1>
            <p>Here's what to focus on each day this week:</p>
            {preview_html}
            <div style="text-align: center; margin: 30px 0;">
                <a href="https://sapframework.com/coaching" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 40px; border-radius: 25px; text-decoration: none; font-weight: bold;">
                    Get Personalized SAP Coaching →
                </a>
            </div>
        </body>
        </html>
        """
        
        return {'subject': subject, 'body': body}


# Example usage
if __name__ == "__main__":
    insights = SAPDailyInsights()
    
    # Generate today's email
    email = insights.generate_daily_email()
    
    print("="*70)
    print("📧 SAP DAILY INSIGHTS EMAIL")
    print("="*70)
    print(f"\nSubject: {email['subject']}")
    print(f"\nStage: {email['stage']}")
    print("\n✅ Email content generated!")
    print("\nTo send:")
    print("1. Set up Mailchimp/ConvertKit/SendGrid")
    print("2. Import subscriber list")
    print("3. Schedule this script to run daily")
    print("4. Emails sent automatically!")
    
    # Save to file for preview
    with open('daily_email_preview.html', 'w') as f:
        f.write(email['body'])
    print("\n📄 Preview saved to: daily_email_preview.html")
