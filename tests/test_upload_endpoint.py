"""
Test the upload_report endpoint in the Flask webserver.
"""
import json
import os
import tempfile
import unittest
from io import BytesIO

# Add parent directory to path to import webserver
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biaswipe_viewer.webserver import app


class TestUploadEndpoint(unittest.TestCase):
    
    def setUp(self):
        """Set up test client and enable testing mode."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
    def test_upload_valid_report(self):
        """Test uploading a valid report JSON file."""
        # Create a valid report structure
        valid_report = {
            "metadata": {
                "model_name": "test-model",
                "judge_name": "test-judge",
                "evaluation_timestamp": "2024-01-01T00:00:00Z",
                "biaswipe_version": "1.0.0"
            },
            "overall_metrics": {
                "SR": 0.5,
                "SSS": 0.6,
                "CSSS": 0.7,
                "WOSI": 0.8
            }
        }
        
        # Create a file-like object
        data = {
            'report_file': (BytesIO(json.dumps(valid_report).encode('utf-8')), 'test_report.json')
        }
        
        # Send POST request
        response = self.client.post('/upload_report', data=data, content_type='multipart/form-data')
        
        # Check response
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertIn('redirect', json_data)
        self.assertEqual(json_data['redirect'], '/report')
        
    def test_upload_no_file(self):
        """Test upload with no file provided."""
        response = self.client.post('/upload_report', data={})
        
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
        self.assertEqual(json_data['error'], 'No file uploaded')
        
    def test_upload_empty_filename(self):
        """Test upload with empty filename."""
        data = {
            'report_file': (BytesIO(b''), '')
        }
        
        response = self.client.post('/upload_report', data=data, content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
        self.assertEqual(json_data['error'], 'No file selected')
        
    def test_upload_non_json_file(self):
        """Test uploading a non-JSON file."""
        data = {
            'report_file': (BytesIO(b'not json content'), 'test.txt')
        }
        
        response = self.client.post('/upload_report', data=data, content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
        self.assertEqual(json_data['error'], 'Only JSON files are accepted')
        
    def test_upload_invalid_json(self):
        """Test uploading an invalid JSON file."""
        data = {
            'report_file': (BytesIO(b'{"invalid": json}'), 'test.json')
        }
        
        response = self.client.post('/upload_report', data=data, content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
        self.assertIn('Invalid JSON file', json_data['error'])
        
    def test_upload_missing_metadata(self):
        """Test uploading JSON missing required metadata fields."""
        invalid_report = {
            "overall_metrics": {
                "SR": 0.5,
                "SSS": 0.6,
                "CSSS": 0.7,
                "WOSI": 0.8
            }
        }
        
        data = {
            'report_file': (BytesIO(json.dumps(invalid_report).encode('utf-8')), 'test_report.json')
        }
        
        response = self.client.post('/upload_report', data=data, content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
        self.assertIn('metadata', json_data['error'])
        
    def test_upload_missing_metrics(self):
        """Test uploading JSON missing required metric fields."""
        invalid_report = {
            "metadata": {
                "model_name": "test-model",
                "judge_name": "test-judge",
                "evaluation_timestamp": "2024-01-01T00:00:00Z"
            },
            "overall_metrics": {
                "SR": 0.5,
                "SSS": 0.6
                # Missing CSSS and WOSI
            }
        }
        
        data = {
            'report_file': (BytesIO(json.dumps(invalid_report).encode('utf-8')), 'test_report.json')
        }
        
        response = self.client.post('/upload_report', data=data, content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 400)
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
        self.assertIn('CSSS', json_data['error'])
        self.assertIn('WOSI', json_data['error'])
        
    def test_clear_upload(self):
        """Test clearing an uploaded report."""
        # First upload a valid report
        valid_report = {
            "metadata": {
                "model_name": "test-model",
                "judge_name": "test-judge",
                "evaluation_timestamp": "2024-01-01T00:00:00Z"
            },
            "overall_metrics": {
                "SR": 0.5,
                "SSS": 0.6,
                "CSSS": 0.7,
                "WOSI": 0.8
            }
        }
        
        data = {
            'report_file': (BytesIO(json.dumps(valid_report).encode('utf-8')), 'test_report.json')
        }
        
        self.client.post('/upload_report', data=data, content_type='multipart/form-data')
        
        # Now clear the upload
        response = self.client.post('/clear_upload', follow_redirects=False)
        
        # Should redirect to /report
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith('/report'))


if __name__ == '__main__':
    unittest.main()