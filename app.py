from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import requests
import re
from urllib.parse import urlparse
import ssl
import socket
from datetime import datetime
import whois
import dns.resolver
import traceback
import logging

# Import our ML model
from ml_model import PhishingMLModel

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridPhishingDetector:
    def __init__(self):
        # Rule-based detection parameters
        self.suspicious_keywords = [
            'verify', 'suspend', 'urgent', 'immediate', 'confirm',
            'update', 'secure', 'account', 'login', 'signin',
            'banking', 'paypal', 'amazon', 'microsoft', 'google'
        ]
        
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
        
        # Initialize ML model
        print("🤖 Initializing ML model...")
        try:
            self.ml_model = PhishingMLModel()
            self.ml_model.load_model()
            print("✅ ML model initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing ML model: {e}")
            self.ml_model = None
        
    def analyze_url(self, url):
        """Hybrid analysis combining rule-based and ML approaches"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            parsed_url = urlparse(url)
            
            # Rule-based analysis
            rule_score, rule_warnings = self._rule_based_analysis(parsed_url, url)
            
            # ML-based analysis (if available)
            ml_score = 0
            ml_result = {'probability': 0, 'confidence': 'low'}
            
            if self.ml_model and self.ml_model.is_trained:
                try:
                    ml_result = self.ml_model.predict(url)
                    ml_score = ml_result['probability'] * 100
                except Exception as e:
                    print(f"⚠️ ML prediction failed: {e}")
                    ml_result = {'probability': 0, 'confidence': 'low'}
            
            # Combine scores (weighted average: 70% rule-based, 30% ML if available)
            if self.ml_model and self.ml_model.is_trained:
                combined_score = (rule_score * 0.7) + (ml_score * 0.3)
            else:
                combined_score = rule_score
            
            # Determine risk level
            if combined_score >= 70:
                risk_level = "HIGH"
            elif combined_score >= 40:
                risk_level = "MEDIUM"
            elif combined_score >= 20:
                risk_level = "LOW"
            else:
                risk_level = "SAFE"
            
            # Add ML insights to warnings if available
            if self.ml_model and self.ml_model.is_trained:
                if ml_result['probability'] > 0.7:
                    rule_warnings.append(f"🤖 ML model: HIGH phishing probability ({ml_result['probability']:.3f})")
                elif ml_result['probability'] > 0.4:
                    rule_warnings.append(f"🤖 ML model: MODERATE phishing risk ({ml_result['probability']:.3f})")
                else:
                    rule_warnings.append(f"🤖 ML model: LOW phishing risk ({ml_result['probability']:.3f})")
                
                rule_warnings.append(f"🔍 Confidence: {ml_result['confidence'].upper()}")
                rule_warnings.append(f"📊 Analysis: Hybrid (Rule-based + Machine Learning)")
            else:
                rule_warnings.append(f"📊 Analysis: Rule-based only (ML model not available)")
            
            return {
                'url': url,
                'risk_level': risk_level,
                'risk_score': int(combined_score),
                'rule_based_score': int(rule_score),
                'ml_score': int(ml_score),
                'ml_probability': round(ml_result['probability'], 3),
                'ml_confidence': ml_result['confidence'],
                'warnings': rule_warnings,
                'is_phishing': combined_score >= 40,
                'analysis_method': 'Hybrid (Rule-based + ML)' if self.ml_model and self.ml_model.is_trained else 'Rule-based only',
                'ml_features': ml_result.get('features', {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing URL {url}: {e}")
            logger.error(traceback.format_exc())
            return {
                'url': url,
                'risk_level': "ERROR",
                'risk_score': 0,
                'rule_based_score': 0,
                'ml_score': 0,
                'ml_probability': 0,
                'warnings': [f"❌ Error analyzing URL: {str(e)}"],
                'is_phishing': False,
                'analysis_method': 'Error'
            }
    
    def _rule_based_analysis(self, parsed_url, full_url):
        """Enhanced rule-based analysis"""
        risk_score = 0
        warnings = []
        
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()
        
        # Check suspicious keywords in domain
        for keyword in self.suspicious_keywords:
            if keyword in domain:
                risk_score += 15
                warnings.append(f"🚨 Suspicious keyword '{keyword}' in domain")
        
        # Check suspicious TLDs
        for tld in self.suspicious_tlds:
            if domain.endswith(tld):
                risk_score += 20
                warnings.append(f"⚠️ Suspicious top-level domain: {tld}")
        
        # Check for IP address instead of domain
        if re.match(r'\d+\.\d+\.\d+\.\d+', domain):
            risk_score += 30
            warnings.append("🔍 URL uses IP address instead of domain name")
        
        # Check subdomain count
        subdomain_count = domain.count('.')
        if subdomain_count > 3:
            risk_score += 15
            warnings.append("📊 Excessive number of subdomains detected")
        elif subdomain_count > 2:
            risk_score += 5
            warnings.append("📊 Multiple subdomains detected")
        
        # Check URL shorteners
        shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'short.link']
        if any(shortener in domain for shortener in shorteners):
            risk_score += 10
            warnings.append("🔗 URL shortener detected")
        
        # Check for suspicious characters in domain
        if any(char in domain for char in ['-', '_']) and len(domain.split('.')[0]) > 15:
            risk_score += 10
            warnings.append("🔤 Suspicious characters in long domain name")
        
        # Check for excessive hyphens
        if domain.count('-') > 2:
            risk_score += 10
            warnings.append("➖ Excessive hyphens in domain name")
        
        # Check URL length
        if len(full_url) > 100:
            risk_score += 10
            warnings.append("📏 Unusually long URL detected")
        elif len(full_url) > 200:
            risk_score += 20
            warnings.append("📏 Extremely long URL detected")
        
        # Check for HTTPS
        if not full_url.startswith('https://'):
            risk_score += 5
            warnings.append("🔒 URL does not use HTTPS encryption")
        
        # Check for suspicious path patterns
        suspicious_paths = ['login', 'signin', 'verify', 'update', 'confirm', 'secure']
        for sus_path in suspicious_paths:
            if sus_path in path:
                risk_score += 8
                warnings.append(f"🛤️ Suspicious path pattern: '{sus_path}'")
        
        # Check for URL parameters that might indicate phishing
        if '?' in full_url:
            query_params = parsed_url.query.lower()
            suspicious_params = ['redirect', 'url', 'next', 'return', 'continue']
            for param in suspicious_params:
                if param in query_params:
                    risk_score += 8
                    warnings.append(f"🔗 Suspicious URL parameter: '{param}'")
        
        # Check domain age using WHOIS (with error handling)
        try:
            domain_info = whois.whois(domain)
            if domain_info.creation_date:
                creation_date = domain_info.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                
                if creation_date:
                    age_days = (datetime.now() - creation_date).days
                    if age_days < 30:
                        risk_score += 25
                        warnings.append("📅 Domain is very new (less than 30 days)")
                    elif age_days < 90:
                        risk_score += 15
                        warnings.append("📅 Domain is relatively new (less than 90 days)")
                    elif age_days < 365:
                        risk_score += 5
                        warnings.append("📅 Domain is less than 1 year old")
                    else:
                        warnings.append(f"📅 Domain age: {age_days} days (established)")
        except Exception as e:
            warnings.append("❓ Could not retrieve domain registration information")
        
        # Check SSL certificate (basic check)
        try:
            if full_url.startswith('https://'):
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                        if cert:
                            warnings.append("🔐 Valid SSL certificate detected")
                        else:
                            risk_score += 15
                            warnings.append("🚫 Invalid or missing SSL certificate")
        except Exception as e:
            if full_url.startswith('https://'):
                risk_score += 10
                warnings.append("⚠️ Could not verify SSL certificate")
        
        # Check for common phishing indicators in full URL
        phishing_indicators = [
            'urgent', 'immediate', 'suspend', 'expire', 'limited',
            'action-required', 'verify-now', 'click-here'
        ]
        for indicator in phishing_indicators:
            if indicator in full_url.lower():
                risk_score += 12
                warnings.append(f"🚨 Phishing indicator detected: '{indicator}'")
        
        # Check for brand impersonation patterns
        brand_patterns = [
            r'paypal.*verify', r'amazon.*security', r'microsoft.*update',
            r'google.*alert', r'apple.*locked', r'facebook.*security'
        ]
        for pattern in brand_patterns:
            if re.search(pattern, full_url.lower()):
                risk_score += 20
                warnings.append("🏢 Potential brand impersonation detected")
        
        return min(risk_score, 100), warnings  # Cap at 100

# Initialize the detector
try:
    detector = HybridPhishingDetector()
    logger.info("✅ Hybrid detector initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize detector: {e}")
    detector = None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('.', filename)

# FIXED: Add the /api/check route that your frontend expects
@app.route('/api/check', methods=['POST'])
def api_check():
    """API endpoint for URL checking - matches frontend expectation"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'error': 'URL is required',
                'success': False
            }), 400
        
        logger.info(f"🔍 API check for URL: {url}")
        
        if detector is None:
            return jsonify({
                'error': 'Detector not initialized',
                'success': False
            }), 500
        
        # Perform analysis
        result = detector.analyze_url(url)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"❌ API check error: {e}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'success': False
        }), 500

# Keep the original /analyze route for backward compatibility
@app.route('/analyze', methods=['POST'])
def analyze_url():
    """Original analyze URL endpoint"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'error': 'URL is required',
                'success': False
            }), 400
        
        if detector is None:
            return jsonify({
                'error': 'Detector not initialized',
                'success': False
            }), 500
        
        # Perform hybrid analysis
        result = detector.analyze_url(url)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"❌ Analyze error: {e}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'success': False
        }), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple URLs at once"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls or not isinstance(urls, list):
            return jsonify({
                'error': 'URLs list is required',
                'success': False
            }), 400
        
        if len(urls) > 10:  # Limit batch size
            return jsonify({
                'error': 'Maximum 10 URLs allowed per batch',
                'success': False
            }), 400
        
        if detector is None:
            return jsonify({
                'error': 'Detector not initialized',
                'success': False
            }), 500
        
        results = []
        for url in urls:
            if url.strip():
                result = detector.analyze_url(url.strip())
                results.append(result)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_analyzed': len(results)
        })
        
    except Exception as e:
        logger.error(f"❌ Batch analyze error: {e}")
        return jsonify({
            'error': f'Batch analysis failed: {str(e)}',
            'success': False
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the ML model"""
    try:
        model_info = detector.ml_model.get_model_info()
        return jsonify({
            'success': True,
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to get model info: {str(e)}',
            'success': False
        }), 500

@app.route('/retrain-model', methods=['POST'])
def retrain_model():
    """Retrain the ML model with current feature set"""
    try:
        if detector is None or detector.ml_model is None:
            return jsonify({
                'error': 'Detector not initialized',
                'success': False
            }), 500
        
        logger.info("🔄 Starting model retraining...")
        
        # Clear existing model
        detector.ml_model.model = None
        detector.ml_model.is_trained = False
        
        # Retrain with current feature extraction
        success = detector.ml_model.train_model()
        
        if success:
            logger.info("✅ Model retrained successfully")
            return jsonify({
                'success': True,
                'message': 'Model retrained successfully',
                'feature_count': len(detector.ml_model.feature_names) if detector.ml_model.feature_names else 0
            })
        else:
            return jsonify({
                'error': 'Model retraining failed',
                'success': False
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Retraining error: {e}")
        return jsonify({
            'error': f'Retraining failed: {str(e)}',
            'success': False
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'detector_initialized': detector is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if detector and detector.ml_model:
            status['ml_model_trained'] = detector.ml_model.is_trained
        else:
            status['ml_model_trained'] = False
            
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/debug-features', methods=['POST'])
def debug_features():
    """Debug feature extraction"""
    try:
        data = request.get_json()
        url = data.get('url', 'https://google.com')
        
        if detector and detector.ml_model:
            features = detector.ml_model.extract_features(url)
            return jsonify({
                'url': url,
                'feature_count': len(features),
                'expected_count': 38,
                'features': features,
                'model_trained': detector.ml_model.is_trained
            })
        else:
            return jsonify({'error': 'Detector not available'})
            
    except Exception as e:
        return jsonify({'error': str(e)})


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False,
        'available_endpoints': [
            'POST /api/check',
            'POST /analyze', 
            'POST /batch-analyze',
            'GET /health'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Hybrid Phishing Detection Server...")
    print("🤖 ML Model Status:", "Trained" if detector and detector.ml_model and detector.ml_model.is_trained else "Not Trained")
    print("🌐 Server running on http://localhost:5000")
    print("📊 Available endpoints:")
    print("   - POST /api/check - Check single URL (for frontend)")
    print("   - POST /analyze - Analyze single URL")
    print("   - POST /batch-analyze - Analyze multiple URLs")
    print("   - GET /health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)