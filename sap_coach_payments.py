"""
SAP COACH - Payment Integration
Stripe subscription management

Pricing:
- Free Trial: 10 messages
- Pro: $29/month
- Team: $99/month (up to 5 users)
"""

import stripe
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

class SubscriptionManager:
    """Manages user subscriptions and payment"""
    
    # Pricing (in cents)
    PLANS = {
        'free': {
            'name': 'Free Trial',
            'price': 0,
            'message_limit': 10,
            'features': ['10 messages', 'Basic SAP diagnosis', 'Email support']
        },
        'pro': {
            'name': 'Pro',
            'price': 2900,  # $29.00
            'message_limit': None,  # Unlimited
            'stripe_price_id': 'price_xxx',  # Replace with actual Stripe price ID
            'features': ['Unlimited messages', 'Advanced SAP insights', 'Priority support', 'Conversation history']
        },
        'team': {
            'name': 'Team',
            'price': 9900,  # $99.00
            'message_limit': None,
            'stripe_price_id': 'price_yyy',  # Replace with actual Stripe price ID
            'features': ['Everything in Pro', 'Up to 5 team members', 'Team analytics', 'API access']
        }
    }
    
    def __init__(self):
        self.users = {}  # In production, use database
    
    def create_checkout_session(self, plan: str, user_email: str, success_url: str, cancel_url: str) -> Dict:
        """
        Create Stripe checkout session
        
        Args:
            plan: 'pro' or 'team'
            user_email: User's email
            success_url: Where to redirect after success
            cancel_url: Where to redirect if cancelled
            
        Returns:
            Dict with checkout session URL
        """
        if plan not in ['pro', 'team']:
            raise ValueError("Invalid plan. Must be 'pro' or 'team'")
        
        plan_info = self.PLANS[plan]
        
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': plan_info['stripe_price_id'],
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                customer_email=user_email,
                metadata={
                    'plan': plan
                }
            )
            
            return {
                'success': True,
                'checkout_url': session.url,
                'session_id': session.id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_subscription_status(self, user_id: str) -> Dict:
        """
        Check if user has active subscription
        
        Returns:
            Dict with subscription status and limits
        """
        # In production, query database
        user = self.users.get(user_id, {
            'plan': 'free',
            'messages_used': 0,
            'subscription_end': None
        })
        
        plan = user.get('plan', 'free')
        plan_info = self.PLANS[plan]
        
        messages_used = user.get('messages_used', 0)
        message_limit = plan_info['message_limit']
        
        # Check if limit reached
        limit_reached = message_limit is not None and messages_used >= message_limit
        
        # Check if subscription expired
        subscription_end = user.get('subscription_end')
        is_expired = False
        if subscription_end:
            is_expired = datetime.fromisoformat(subscription_end) < datetime.now()
        
        return {
            'plan': plan,
            'plan_name': plan_info['name'],
            'messages_used': messages_used,
            'message_limit': message_limit,
            'limit_reached': limit_reached,
            'is_expired': is_expired,
            'can_chat': not limit_reached and not is_expired,
            'features': plan_info['features']
        }
    
    def increment_message_count(self, user_id: str):
        """Increment user's message count"""
        if user_id not in self.users:
            self.users[user_id] = {
                'plan': 'free',
                'messages_used': 0,
                'subscription_end': None
            }
        
        self.users[user_id]['messages_used'] += 1
    
    def upgrade_user(self, user_id: str, plan: str, subscription_id: str):
        """Upgrade user to paid plan"""
        self.users[user_id] = {
            'plan': plan,
            'messages_used': 0,
            'subscription_id': subscription_id,
            'subscription_end': (datetime.now() + timedelta(days=30)).isoformat()
        }
    
    def handle_webhook(self, event: Dict) -> Dict:
        """
        Handle Stripe webhook events
        
        Events to handle:
        - checkout.session.completed: User subscribed
        - customer.subscription.deleted: User cancelled
        - invoice.payment_failed: Payment failed
        """
        event_type = event['type']
        
        if event_type == 'checkout.session.completed':
            session = event['data']['object']
            user_email = session['customer_email']
            plan = session['metadata']['plan']
            subscription_id = session['subscription']
            
            # Upgrade user (in production, use email to find user_id)
            user_id = user_email  # Simplified
            self.upgrade_user(user_id, plan, subscription_id)
            
            return {'status': 'user_upgraded', 'user_id': user_id, 'plan': plan}
        
        elif event_type == 'customer.subscription.deleted':
            subscription = event['data']['object']
            subscription_id = subscription['id']
            
            # Downgrade user to free (in production, find user by subscription_id)
            # self.downgrade_user(user_id)
            
            return {'status': 'subscription_cancelled'}
        
        elif event_type == 'invoice.payment_failed':
            invoice = event['data']['object']
            customer_email = invoice['customer_email']
            
            # Send payment failed email
            return {'status': 'payment_failed', 'email': customer_email}
        
        return {'status': 'unhandled_event', 'type': event_type}


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("💳 SAP COACH - Payment Integration")
    print("="*70)
    
    # Check for Stripe key
    if not os.getenv("STRIPE_SECRET_KEY"):
        print("\n❌ STRIPE_SECRET_KEY not set")
        print("\nTo use Stripe:")
        print("1. Create account at https://stripe.com")
        print("2. Get API keys from Dashboard")
        print("3. Create products and prices")
        print("4. Set environment variable:")
        print("   set STRIPE_SECRET_KEY=sk_test_...")
        exit(1)
    
    manager = SubscriptionManager()
    
    # Test subscription check
    print("\n📊 Testing subscription status:")
    status = manager.check_subscription_status("test_user")
    print(f"  Plan: {status['plan_name']}")
    print(f"  Messages used: {status['messages_used']}/{status['message_limit'] or '∞'}")
    print(f"  Can chat: {status['can_chat']}")
    print(f"  Features: {', '.join(status['features'])}")
    
    # Test message increment
    print("\n📈 Simulating message usage:")
    for i in range(12):
        manager.increment_message_count("test_user")
        status = manager.check_subscription_status("test_user")
        print(f"  Message {i+1}: Can chat = {status['can_chat']}")
        if not status['can_chat']:
            print("  ⚠️ Limit reached! Upgrade required.")
            break
    
    print("\n✅ Payment integration ready!")
    print("\nNext steps:")
    print("1. Create Stripe account")
    print("2. Create products (Pro $29, Team $99)")
    print("3. Get price IDs and update PLANS dict")
    print("4. Set up webhook endpoint")
    print("5. Test checkout flow")
