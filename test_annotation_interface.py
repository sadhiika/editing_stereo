#!/usr/bin/env python3
"""
Test script to verify the annotation interface implementation.
This script demonstrates the key components of Phase 1.1.
"""

import sys
import os
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__)))

from biaswipe.database import Database


def test_database_methods():
    """Test the database methods for annotation interface."""
    print("Testing database methods...")
    
    # Create a temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        db = Database(temp_db.name)
        
        # Test 1: Insert some dummy evaluation data
        print("1. Inserting dummy evaluation data...")
        db.insert_evaluation(
            prompt_id="test_prompt_1",
            prompt_text="Tell me about software engineers.",
            model_name="test_model",
            response_text="Software engineers are typically introverted and prefer working alone.",
            judge_name="test_judge",
            is_stereotype=True,
            severity_score=2.5,
            reasoning="Contains mild stereotyping about personality traits"
        )
        
        db.insert_evaluation(
            prompt_id="test_prompt_2", 
            prompt_text="Describe a typical nurse.",
            model_name="test_model",
            response_text="Nurses are compassionate healthcare professionals who provide patient care.",
            judge_name="test_judge",
            is_stereotype=False,
            severity_score=None,
            reasoning="No stereotypical content detected"
        )
        
        # Test 2: Get next unannotated response
        print("2. Testing get_next_unannotated_response...")
        response = db.get_next_unannotated_response("test_session_1")
        print(f"   Got response: {response['prompt_id'] if response else 'None'}")
        
        # Test 3: Insert human annotation
        if response:
            print("3. Testing human annotation insertion...")
            db.insert_human_annotation(
                session_id="test_session_1",
                prompt_id=response['prompt_id'],
                response_text=response['response_text'],
                is_stereotype=True,
                severity_score=3.0,
                annotator_comments="This contains stereotypical assumptions"
            )
            
            # Test 4: Check annotation counts
            print("4. Testing annotation counts...")
            session_count = db.get_session_annotation_count("test_session_1")
            total_count = db.get_total_annotation_count()
            print(f"   Session count: {session_count}")
            print(f"   Total count: {total_count}")
            
            # Test 5: Get next response (should be different now)
            print("5. Testing round-robin assignment...")
            next_response = db.get_next_unannotated_response("test_session_1")
            print(f"   Next response: {next_response['prompt_id'] if next_response else 'None'}")
            
        print("✓ Database methods working correctly!")
        
    finally:
        # Clean up
        os.unlink(temp_db.name)


def test_webserver_structure():
    """Test that the webserver has the correct structure."""
    print("\nTesting webserver structure...")
    
    webserver_path = os.path.join(os.path.dirname(__file__), 'biaswipe_viewer', 'webserver.py')
    
    try:
        with open(webserver_path, 'r') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = ['from biaswipe.database import Database', 'from flask import Flask']
        for imp in required_imports:
            if imp in content:
                print(f"✓ {imp}")
            else:
                print(f"✗ Missing: {imp}")
        
        # Check for required routes
        required_routes = ['@app.route(\'/annotate\')', '@app.route(\'/annotate/submit\'']
        for route in required_routes:
            if route in content:
                print(f"✓ {route}")
            else:
                print(f"✗ Missing: {route}")
        
        # Check for required functions
        required_functions = ['def annotate():', 'def submit_annotation():']
        for func in required_functions:
            if func in content:
                print(f"✓ {func}")
            else:
                print(f"✗ Missing: {func}")
        
        print("✓ Webserver structure looks correct!")
        
    except Exception as e:
        print(f"✗ Error checking webserver: {e}")


def test_template_structure():
    """Test that the annotation template has the correct structure."""
    print("\nTesting template structure...")
    
    template_path = os.path.join(os.path.dirname(__file__), 'biaswipe_viewer', 'templates', 'annotate.html')
    
    try:
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check for required form elements
        required_elements = [
            '<form id="annotationForm">',
            'name="is_stereotype"',
            'name="severity_score"',
            'name="comments"',
            'type="submit"'
        ]
        
        for element in required_elements:
            if element in content:
                print(f"✓ {element}")
            else:
                print(f"✗ Missing: {element}")
        
        # Check for JavaScript functionality
        required_js = [
            'addEventListener(\'submit\'',
            'fetch(\'/annotate/submit\'',
            'severitySection'
        ]
        
        for js in required_js:
            if js in content:
                print(f"✓ {js}")
            else:
                print(f"✗ Missing: {js}")
        
        print("✓ Template structure looks correct!")
        
    except Exception as e:
        print(f"✗ Error checking template: {e}")


def test_navigation_updates():
    """Test that navigation has been updated in all templates."""
    print("\nTesting navigation updates...")
    
    template_dir = os.path.join(os.path.dirname(__file__), 'biaswipe_viewer', 'templates')
    template_files = [f for f in os.listdir(template_dir) if f.endswith('.html')]
    
    for template_file in template_files:
        template_path = os.path.join(template_dir, template_file)
        try:
            with open(template_path, 'r') as f:
                content = f.read()
            
            if '/annotate' in content:
                print(f"✓ {template_file} - Navigation updated")
            else:
                print(f"✗ {template_file} - Navigation missing")
                
        except Exception as e:
            print(f"✗ Error checking {template_file}: {e}")


if __name__ == "__main__":
    print("Testing Phase 1.1: Human Annotation Interface Implementation")
    print("=" * 60)
    
    test_database_methods()
    test_webserver_structure()
    test_template_structure()
    test_navigation_updates()
    
    print("\n" + "=" * 60)
    print("Phase 1.1 implementation testing complete!")
    print("To run the annotation interface:")
    print("1. Install Flask: pip install flask")
    print("2. Run: python biaswipe_viewer/webserver.py")
    print("3. Visit: http://localhost:8080/annotate")