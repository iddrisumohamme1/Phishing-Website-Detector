import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re
from urllib.parse import urlparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class PhishingMLModel:
    def __init__(self, csv_file_path='PhiUSIIL_Phishing_URL_Dataset.csv'):
        """
        Initialize the ML model with CSV dataset support
        
        Args:
            csv_file_path (str): Path to the CSV file containing URL data
        """
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.is_trained = False
        self.csv_file_path = csv_file_path
        
        # Model file paths
        self.model_path = 'models/phishing_model.pkl'
        self.vectorizer_path = 'models/vectorizer.pkl'
        self.feature_names_path = 'models/feature_names.pkl'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Check if CSV file exists
        if not os.path.exists(self.csv_file_path):
            print(f"⚠️ CSV file not found: {self.csv_file_path}")
            print("📝 Will use fallback training data if needed")
        else:
            print(f"✅ CSV dataset found: {self.csv_file_path}")
    
    def load_csv_data(self):
        """
        Load and process the CSV dataset
        
        Returns:
            tuple: (urls, labels) or None if failed
        """
        try:
            print(f"📂 Loading CSV dataset: {self.csv_file_path}")
            
            # Try different common CSV formats
            possible_separators = [',', ';', '\t']
            df = None
            
            for sep in possible_separators:
                try:
                    df = pd.read_csv(self.csv_file_path, separator=sep)
                    if len(df.columns) >= 2:  # At least URL and label columns
                        print(f"✅ Successfully loaded CSV with separator: '{sep}'")
                        break
                except:
                    continue
            
            if df is None:
                print("❌ Could not read CSV file with any common separator")
                return None
            
            print(f"📊 Dataset shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Display first few rows to understand structure
            print("\n🔍 First 3 rows of dataset:")
            print(df.head(3))
            
            # Auto-detect URL and label columns
            url_column = None
            label_column = None
            
            # Common column names for URLs
            url_candidates = ['url', 'URL', 'link', 'website', 'domain', 'address']
            for col in df.columns:
                if any(candidate.lower() in col.lower() for candidate in url_candidates):
                    url_column = col
                    break
            
            # If not found, assume first column is URL
            if url_column is None:
                url_column = df.columns[0]
                print(f"🔍 Assuming first column '{url_column}' contains URLs")
            
            # Common column names for labels
            label_candidates = ['label', 'class', 'type', 'category', 'phishing', 'legitimate', 'target']
            for col in df.columns:
                if any(candidate.lower() in col.lower() for candidate in label_candidates):
                    label_column = col
                    break
            
            # If not found, assume last column is label
            if label_column is None:
                label_column = df.columns[-1]
                print(f"🔍 Assuming last column '{label_column}' contains labels")
            
            print(f"📍 Using URL column: '{url_column}'")
            print(f"🏷️ Using label column: '{label_column}'")
            
            # Extract URLs and labels
            urls = df[url_column].astype(str).tolist()
            labels = df[label_column].tolist()
            
            # Clean URLs (remove NaN, empty strings)
            clean_data = []
            for url, label in zip(urls, labels):
                if pd.notna(url) and pd.notna(label) and str(url).strip() and str(url) != 'nan':
                    clean_data.append((str(url).strip(), label))
            
            if not clean_data:
                print("❌ No valid URL-label pairs found in dataset")
                return None
            
            urls, labels = zip(*clean_data)
            urls = list(urls)
            labels = list(labels)
            
            # Normalize labels to 0 (legitimate) and 1 (phishing)
            normalized_labels = []
            unique_labels = set(labels)
            print(f"🏷️ Unique labels found: {unique_labels}")
            
            # Auto-detect label format
            if all(isinstance(label, (int, float)) for label in labels):
                # Numeric labels
                for label in labels:
                    if label in [0, 1]:
                        normalized_labels.append(int(label))
                    elif label > 0:
                        normalized_labels.append(1)  # Assume positive = phishing
                    else:
                        normalized_labels.append(0)  # Assume 0/negative = legitimate
            else:
                # Text labels
                for label in labels:
                    label_str = str(label).lower().strip()
                    if label_str in ['phishing', 'malicious', 'bad', '1', 'true', 'positive']:
                        normalized_labels.append(1)
                    elif label_str in ['legitimate', 'good', 'safe', '0', 'false', 'negative']:
                        normalized_labels.append(0)
                    else:
                        # Try to guess based on common patterns
                        if 'phish' in label_str or 'malicious' in label_str or 'bad' in label_str:
                            normalized_labels.append(1)
                        elif 'legit' in label_str or 'good' in label_str or 'safe' in label_str:
                            normalized_labels.append(0)
                        else:
                            # Default to legitimate if unsure
                            normalized_labels.append(0)
                            print(f"⚠️ Unknown label '{label_str}', defaulting to legitimate")
            
            # Verify we have both classes
            unique_normalized = set(normalized_labels)
            legitimate_count = normalized_labels.count(0)
            phishing_count = normalized_labels.count(1)
            
            print(f"📊 Dataset statistics:")
            print(f"   Total URLs: {len(urls)}")
            print(f"   Legitimate URLs: {legitimate_count}")
            print(f"   Phishing URLs: {phishing_count}")
            print(f"   Classes found: {unique_normalized}")
            
            if len(unique_normalized) < 2:
                print("⚠️ Warning: Only one class found in dataset")
            
            if legitimate_count == 0 or phishing_count == 0:
                print("⚠️ Warning: Imbalanced dataset - one class is missing")
            
            return urls, normalized_labels
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            print("📝 Will use fallback training data")
            return None
    
    def create_training_data(self):
        """Create training data from CSV file or fallback data"""
        
        # Try to load from CSV first
        csv_data = self.load_csv_data()
        if csv_data is not None:
            return csv_data
        
        print("📝 Using fallback training data...")
        
        # Fallback to hardcoded data if CSV fails
        legitimate_urls = [
            'https://www.google.com', 'https://www.facebook.com', 'https://www.amazon.com',
            'https://www.microsoft.com', 'https://www.apple.com', 'https://www.netflix.com',
            'https://www.youtube.com', 'https://www.twitter.com', 'https://www.instagram.com',
            'https://www.linkedin.com', 'https://www.github.com', 'https://www.stackoverflow.com',
            'https://www.wikipedia.org', 'https://www.reddit.com', 'https://www.ebay.com',
            'https://www.paypal.com', 'https://www.adobe.com', 'https://www.salesforce.com',
            'https://www.oracle.com', 'https://www.ibm.com', 'https://www.cisco.com',
            'https://www.intel.com', 'https://www.nvidia.com', 'https://www.amd.com',
            'https://www.dell.com', 'https://www.hp.com', 'https://www.lenovo.com',
            'https://www.samsung.com', 'https://www.sony.com', 'https://www.lg.com',
            'https://www.cnn.com', 'https://www.bbc.com', 'https://www.nytimes.com',
            'https://www.washingtonpost.com', 'https://www.reuters.com', 'https://www.bloomberg.com',
            'https://www.forbes.com', 'https://www.techcrunch.com', 'https://www.wired.com',
            'https://www.theverge.com', 'https://www.engadget.com', 'https://www.ars-technica.com'
        ]
        
        phishing_urls = [
            'http://paypal-verify.tk/login', 'https://amazon-security.ml/update',
            'http://192.168.1.100/microsoft-login', 'https://bit.ly/urgent-paypal-verify',
            'http://google-alert.ga/suspended', 'https://apple-locked.cf/unlock',
            'http://facebook-security.tk/verify', 'https://netflix-billing.ml/update',
            'http://instagram-verify.ga/account', 'https://twitter-suspended.cf/appeal',
            'http://linkedin-security.tk/confirm', 'https://github-alert.ml/verify',
            'http://microsoft-update.ga/security', 'https://adobe-license.cf/renew',
            'http://paypal.verification-required.tk', 'https://amazon.account-locked.ml',
            'http://apple.com-verification.ga', 'https://google.com-alert.cf',
            'http://facebook.com-security.tk', 'https://netflix.com-billing.ml',
            'http://secure-paypal-verify.tk/login.php', 'https://amazon-customer-service.ml/help',
            'http://microsoft-office-update.ga/download', 'https://apple-icloud-locked.cf/unlock',
            'http://google-drive-share.tk/document', 'https://facebook-photo-tagged.ml/view',
            'http://urgent-paypal-limitation.ga/resolve', 'https://amazon-prime-expired.cf/renew',
            'http://microsoft-security-alert.tk/scan', 'https://apple-payment-failed.ml/update',
            'http://suspicious-login-google.ga/verify', 'https://facebook-account-disabled.cf/appeal',
            'http://paypal-payment-pending.tk/confirm', 'https://amazon-order-cancelled.ml/details',
            'http://microsoft-license-expired.ga/activate', 'https://apple-subscription-cancelled.cf/restore',
            'http://google-account-compromised.tk/secure', 'https://facebook-unusual-activity.ml/review',
            'http://paypal-refund-processing.ga/claim', 'https://amazon-gift-card-won.cf/redeem',
            'http://microsoft-windows-infected.tk/clean', 'https://apple-warranty-expired.ml/extend'
        ]
        
        # Combine datasets
        all_urls = legitimate_urls + phishing_urls
        labels = [0] * len(legitimate_urls) + [1] * len(phishing_urls)  # 0 = legitimate, 1 = phishing
        
        print(f"📊 Fallback dataset: {len(legitimate_urls)} legitimate + {len(phishing_urls)} phishing URLs")
        
        return all_urls, labels
    
    def extract_features(self, url):
        """Extract comprehensive features from URL"""
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Parse URL components (MOVED OUTSIDE the if block)
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            path = parsed_url.path.lower()
            query = parsed_url.query.lower()
            
            # Basic length features
            features = {
                'url_length': len(url),
                'domain_length': len(domain),
                'path_length': len(path),
                'query_length': len(query),
                
                # Character count features
                'dot_count': url.count('.'),
                'hyphen_count': url.count('-'),
                'underscore_count': url.count('_'),
                'slash_count': url.count('/'),
                'question_count': url.count('?'),
                'equal_count': url.count('='),
                'at_count': url.count('@'),
                'ampersand_count': url.count('&'),
                'percent_count': url.count('%'),
                'hash_count': url.count('#'),
                
                # Domain analysis
                'subdomain_count': max(0, domain.count('.') - 1),
                'domain_has_digits': 1 if any(c.isdigit() for c in domain) else 0,
                'domain_digit_ratio': sum(c.isdigit() for c in domain) / len(domain) if len(domain) > 0 else 0,
                
                # Suspicious patterns
                'has_ip_address': 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', domain) else 0,
                'has_port': 1 if re.search(r':[0-9]+', domain) else 0,
                'has_www': 1 if domain.startswith('www.') else 0,
                
                # Protocol features
                'is_https': 1 if url.startswith('https://') else 0,
                'is_http': 1 if url.startswith('http://') else 0,
                
                # Keyword analysis
                'suspicious_keywords_count': self._count_suspicious_keywords(url),
                'phishing_keywords_count': self._count_phishing_keywords(url),
                'brand_keywords_count': self._count_brand_keywords(url),
                
                # Advanced features
                'has_suspicious_tld': self._check_suspicious_tld(domain),
                'is_url_shortener': self._is_url_shortener(domain),
                'url_entropy': self._calculate_entropy(url),
                'domain_entropy': self._calculate_entropy(domain),
                'path_entropy': self._calculate_entropy(path),
                
                # URL structure features
                'has_multiple_subdomains': 1 if domain.count('.') > 2 else 0,
                'has_suspicious_port': 1 if re.search(r':(8080|8000|3000|4444|5555)', url) else 0,
                'has_redirect_params': 1 if any(param in query for param in ['redirect', 'url', 'next', 'goto']) else 0,
                
                # Character ratio features
                'digit_ratio': sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
                'special_char_ratio': sum(1 for c in url if not c.isalnum() and c not in ':/.-_') / len(url) if len(url) > 0 else 0,
                
                # Length ratios
                'domain_to_url_ratio': len(domain) / len(url) if len(url) > 0 else 0,
                'path_to_url_ratio': len(path) / len(url) if len(url) > 0 else 0,
                
                # Additional suspicious patterns
                'has_double_slash_in_path': 1 if '//' in path else 0,
                'has_suspicious_extension': 1 if any(ext in path for ext in ['.exe', '.zip', '.rar', '.bat', '.scr']) else 0,
                'consecutive_dots': max([len(match.group()) for match in re.finditer(r'\.{2,}', url)] + [0]),
                'consecutive_hyphens': max([len(match.group()) for match in re.finditer(r'-{2,}', url)] + [0])
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error extracting features from URL '{url}': {e}")
            # Return default features if extraction fails
            return {f'feature_{i}': 0 for i in range(38)}

    
    def _count_suspicious_keywords(self, url):
        """Count suspicious keywords in URL"""
        suspicious_words = [
            'verify', 'account', 'update', 'confirm', 'secure', 'suspend',
            'urgent', 'immediate', 'action', 'required', 'expire', 'limited',
            'click', 'here', 'now', 'today', 'warning', 'alert', 'notice'
        ]
        url_lower = url.lower()
        return sum(1 for word in suspicious_words if word in url_lower)
    
    def _count_phishing_keywords(self, url):
        """Count phishing-related keywords in URL"""
        phishing_words = [
            'login', 'signin', 'password', 'credential', 'auth', 'validation',
            'verification', 'security', 'alert', 'warning', 'blocked', 'locked',
            'suspended', 'disabled', 'unauthorized', 'breach', 'compromise'
        ]
        url_lower = url.lower()
        return sum(1 for word in phishing_words if word in url_lower)
    
    def _count_brand_keywords(self, url):
        """Count brand impersonation keywords in URL"""
        brand_words = [
            'paypal', 'amazon', 'microsoft', 'google', 'apple', 'facebook',
            'instagram', 'twitter', 'linkedin', 'netflix', 'spotify', 'ebay',
            'bank', 'chase', 'wells', 'citibank', 'americanexpress', 'visa',
            'mastercard', 'discover', 'adobe', 'oracle', 'salesforce'
        ]
        url_lower = url.lower()
        return sum(1 for word in brand_words if word in url_lower)
    
    def _check_suspicious_tld(self, domain):
        """Check if domain uses suspicious top-level domain"""
        suspicious_tlds = [
            '.tk', '.ml', '.ga', '.cf', '.click', '.download', '.zip',
            '.rar', '.exe', '.scr', '.bat', '.com.br', '.co.cc', '.bit'
        ]
        return 1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0
    
    def _is_url_shortener(self, domain):
        """Check if domain is a URL shortener"""
        shorteners = [
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'short.link',
            'tiny.cc', 'is.gd', 'buff.ly', 'adf.ly', 'bl.ink', 'lnkd.in',
            'rebrand.ly', 'cutt.ly', 'shorturl.at'
        ]
        return 1 if any(shortener in domain for shortener in shorteners) else 0
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        
        # Calculate character frequency
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        text_length = len(text)
        entropy = 0
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def train_model(self):
        """Train the machine learning model"""
        print("🚀 Starting ML model training...")
        
        # Get training data (from CSV or fallback)
        urls, labels = self.create_training_data()
        
        if not urls or not labels:
            print("❌ No training data available")
            return False
        
        # Extract features for all URLs
        print("🔍 Extracting features from URLs...")
        features_list = []
        valid_indices = []
        
        for i, url in enumerate(urls):
            try:
                features = self.extract_features(url)
                features_list.append(list(features.values()))
                valid_indices.append(i)
            except Exception as e:
                print(f"⚠️ Skipping URL {i}: {e}")
                continue
        
        if not features_list:
            print("❌ No valid features extracted")
            return False
        
        # Filter labels to match valid features
        valid_labels = [labels[i] for i in valid_indices]
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(valid_labels)
        
        print(f"📊 Training data shape: {X.shape}")
        print(f"📊 Labels distribution: {np.bincount(y)}")
        
        # Store feature names
        sample_features = self.extract_features(urls[0])
        self.feature_names = list(sample_features.keys())
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train the model
        print("🤖 Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Use all CPU cores
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate model performance
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"🎯 Training accuracy: {train_accuracy:.3f}")
        print(f"🎯 Testing accuracy: {test_accuracy:.3f}")
        
        # Print detailed classification report
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred_test, 
                                  target_names=['Legitimate', 'Phishing']))
        
        # Print feature importance
        self._print_feature_importance()
        
        # Save the trained model
        self.save_model()
        
        self.is_trained = True
        print("✅ Model training completed successfully!")
        
        return True
    
    def _print_feature_importance(self):
        """Print the most important features"""
        if self.model is None or self.feature_names is None:
            return
        
        feature_importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 15 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
    
    def predict(self, url):
        """Predict if URL is phishing"""
        if not self.is_trained:
            # Try to load existing model
            if not self.load_model():
                # If no model exists, train one
                print("🤖 No trained model found. Training new model...")
                if not self.train_model():
                    return {
                        'probability': 0.5,
                        'prediction': 0,
                        'confidence': 'low',
                        'status': 'Model training failed'
                    }
        
        try:
            # Extract features
            features = self.extract_features(url)
            X_features = np.array([list(features.values())])
            
            # Get prediction probability
            probabilities = self.model.predict_proba(X_features)[0]
            phishing_probability = probabilities[1]  # Probability of being phishing
            
            # Get binary prediction
            prediction = int(self.model.predict(X_features)[0])
            
            # Determine confidence level
            confidence_score = abs(phishing_probability - 0.5)
            if confidence_score > 0.4:
                confidence = 'high'
            elif confidence_score > 0.2:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'probability': float(phishing_probability),
                'prediction': prediction,
                'confidence': confidence,
                'status': 'Prediction successful',
                'features': features
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'probability': 0.5,
                'prediction': 0,
                'confidence': 'low',
                'status': f'Prediction failed: {str(e)}'
            }
    
    def save_model(self):
        """Save the trained model to disk"""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                print(f"💾 Model saved to {self.model_path}")
            
            if self.vectorizer is not None:
                joblib.dump(self.vectorizer, self.vectorizer_path)
                print(f"💾 Vectorizer saved to {self.vectorizer_path}")
            
            if self.feature_names is not None:
                joblib.dump(self.feature_names, self.feature_names_path)
                print(f"💾 Feature names saved to {self.feature_names_path}")
                
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            if (os.path.exists(self.model_path) and 
                os.path.exists(self.feature_names_path)):
                
                self.model = joblib.load(self.model_path)
                self.feature_names = joblib.load(self.feature_names_path)
                
                if os.path.exists(self.vectorizer_path):
                    self.vectorizer = joblib.load(self.vectorizer_path)
                
                self.is_trained = True
                print("✅ Model loaded successfully from disk")
                return True
            else:
                print("📝 No saved model found, will need to train new model")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_model_info(self):
        """Get information about the current model"""
        info = {
            'is_trained': self.is_trained,
            'csv_file_path': self.csv_file_path,
            'csv_file_exists': os.path.exists(self.csv_file_path),
            'model_file_exists': os.path.exists(self.model_path),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'model_type': 'Random Forest Classifier'
        }
        
        if self.is_trained and self.model:
            info.update({
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'n_features': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 'Unknown'
            })
        
        return info
    
    def retrain_model(self):
        """Retrain the model with fresh data"""
        print("🔄 Retraining model...")
        self.model = None
        self.is_trained = False
        return self.train_model()
    
    def evaluate_url_batch(self, urls):
        """Evaluate multiple URLs at once"""
        results = []
        for url in urls:
            try:
                result = self.predict(url)
                result['url'] = url
                results.append(result)
            except Exception as e:
                results.append({
                    'url': url,
                    'probability': 0.5,
                    'prediction': 0,
                    'confidence': 'low',
                    'status': f'Error: {str(e)}'
                })
        return results

# Testing and usage functions
def test_model_with_csv():
    """Test the model with CSV data"""
    print("🧪 Testing ML Model with CSV Dataset")
    print("=" * 50)
    
    # Initialize model
    ml_model = PhishingMLModel('PhiUSIIL_Phishing_URL_Dataset.csv')
    
    # Check if CSV file exists
    if not os.path.exists(ml_model.csv_file_path):
        print(f"❌ CSV file not found: {ml_model.csv_file_path}")
        print("📝 Please ensure the CSV file is in the same directory as this script")
        return
    
    # Try to load existing model or train new one
    if not ml_model.load_model():
        print("🤖 Training new model...")
        success = ml_model.train_model()
        if not success:
            print("❌ Model training failed")
            return
    
    # Test with sample URLs
    test_urls = [
        'https://www.google.com',
        'https://www.paypal.com',
        'http://paypal-verify.tk/login',
        'https://amazon-security.ml/update',
        'http://192.168.1.100/microsoft-login',
        'https://www.facebook.com',
        'http://suspicious-bank-alert.ga/verify'
    ]
    
    print("\n🔍 Testing with sample URLs:")
    print("-" * 80)
    
    for url in test_urls:
        result = ml_model.predict(url)
        risk_level = "HIGH" if result['probability'] > 0.7 else "MEDIUM" if result['probability'] > 0.4 else "LOW"
        
        print(f"URL: {url}")
        print(f"   Phishing Probability: {result['probability']:.3f}")
        print(f"   Risk Level: {risk_level}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Status: {result['status']}")
        print()
    
    # Show model information
    print("📊 Model Information:")
    info = ml_model.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

def analyze_csv_structure():
    """Analyze the structure of the CSV file"""
    csv_file = 'PhiUSIIL_Phishing_URL_Dataset.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    print("🔍 Analyzing CSV file structure...")
    print("=" * 50)
    
    try:
        # Try different separators
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(csv_file, sep=sep, nrows=5)
                if len(df.columns) >= 2:
                    print(f"✅ Successfully read with separator: '{sep}'")
                    print(f"📊 Shape: {df.shape}")
                    print(f"📋 Columns: {list(df.columns)}")
                    print(f"📝 First 3 rows:")
                    print(df.head(3))
                    print(f"📈 Data types:")
                    print(df.dtypes)
                    break
            except Exception as e:
                continue
        else:
            print("❌ Could not read CSV file with any common separator")
            
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")

def create_sample_csv():
    """Create a sample CSV file for testing"""
    sample_data = {
        'url': [
            'https://www.google.com',
            'https://www.facebook.com',
            'https://www.amazon.com',
            'http://paypal-verify.tk/login',
            'https://amazon-security.ml/update',
            'http://192.168.1.100/microsoft-login',
            'https://www.microsoft.com',
            'http://suspicious-bank.ga/verify',
            'https://www.apple.com',
            'http://fake-paypal.cf/signin'
        ],
        'label': [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]  # 0 = legitimate, 1 = phishing
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_phishing_dataset.csv', index=False)
    print("✅ Sample CSV file created: sample_phishing_dataset.csv")
    print("📊 Sample data:")
    print(df)

if __name__ == "__main__":
    print("🚀 Phishing Detection ML Model")
    print("=" * 50)
    
    # Check if CSV file exists
    csv_file = 'PhiUSIIL_Phishing_URL_Dataset.csv'
    
    if os.path.exists(csv_file):
        print(f"✅ Found CSV dataset: {csv_file}")
        
        # Analyze CSV structure first
        analyze_csv_structure()
        print("\n" + "="*50 + "\n")
        
        # Test the model
        test_model_with_csv()
        
    else:
        print(f"❌ CSV file not found: {csv_file}")
        print("\n🔧 Options:")
        print("1. Place your CSV file in the same directory as this script")
        print("2. Update the csv_file_path in PhishingMLModel initialization")
        print("3. Create a sample CSV for testing")
        
        choice = input("\nCreate sample CSV? (y/n): ").lower().strip()
        if choice == 'y':
            create_sample_csv()
            print("\n🔄 Now you can test with the sample data:")
            print("ml_model = PhishingMLModel('sample_phishing_dataset.csv')")
        
        print("\n📝 Using fallback training data for demonstration...")
        
        # Test with fallback data
        ml_model = PhishingMLModel('nonexistent.csv')  # Will use fallback data
        ml_model.train_model()
        
        # Test prediction
        test_url = 'http://paypal-verify.tk/login'
        result = ml_model.predict(test_url)
        print(f"\n🧪 Test prediction for: {test_url}")
        print(f"   Phishing probability: {result['probability']:.3f}")
        print(f"   Confidence: {result['confidence']}")