const API_BASE_URL = 'http://localhost:5000';

class PhishingDetector {
    constructor() {
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const checkBtn = document.getElementById('checkBtn');
        const urlInput = document.getElementById('urlInput');

        checkBtn.addEventListener('click', () => this.checkURL());
        
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.checkURL();
            }
        });
    }

    async checkURL() {
        const urlInput = document.getElementById('urlInput');
        const url = urlInput.value.trim();

        if (!url) {
            this.showError('Please enter a URL to check');
            return;
        }

        this.showLoading(true);
        this.hideResults();

        try {
            // FIXED: Use correct endpoint that matches app.py
            const response = await fetch(`${API_BASE_URL}/api/check`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // FIXED: Handle the correct response structure
            if (data.success) {
                this.displayResults(data.result);
            } else {
                this.showError(data.error || 'Unknown error occurred');
            }

        } catch (error) {
            console.error('Error checking URL:', error);
            this.showError('Failed to analyze URL. Please check your connection and try again.');
        } finally {
            this.showLoading(false);
        }
    }

    showLoading(show) {
        const loading = document.getElementById('loading');
        if (loading) {
            if (show) {
                loading.classList.remove('hidden');
            } else {
                loading.classList.add('hidden');
            }
        }
    }

    hideResults() {
        const results = document.getElementById('results');
        if (results) {
            results.classList.add('hidden');
        }
    }

    displayResults(result) {
        const results = document.getElementById('results');
        const riskBadge = document.getElementById('riskBadge');
        const analyzedUrl = document.getElementById('analyzedUrl');
        const riskScore = document.getElementById('riskScore');
        const phishingStatus = document.getElementById('phishingStatus');
        const warningsList = document.getElementById('warningsList');
        const recommendationsList = document.getElementById('recommendationsList');

        // FIXED: Handle potential undefined elements
        if (!results) {
            console.error('Results container not found');
            return;
        }

        // Set risk badge
        if (riskBadge) {
            riskBadge.textContent = result.risk_level;
            riskBadge.className = `risk-badge ${result.risk_level.toLowerCase()}`;
        }

        // Set basic info
        if (analyzedUrl) {
            analyzedUrl.textContent = result.url;
        }
        
        if (riskScore) {
            riskScore.textContent = `${result.risk_score}/100`;
        }
        
        if (phishingStatus) {
            phishingStatus.textContent = result.is_phishing ? 'Potentially Dangerous' : 'Appears Safe';
            phishingStatus.className = result.is_phishing ? 'status-danger' : 'status-safe';
        }

        // Display warnings
        if (warningsList) {
            warningsList.innerHTML = '';
            if (result.warnings && result.warnings.length > 0) {
                result.warnings.forEach(warning => {
                    const li = document.createElement('li');
                    li.textContent = warning;
                    warningsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'No specific warnings detected';
                li.style.color = '#28a745';
                warningsList.appendChild(li);
            }
        }

        // Display recommendations
        if (recommendationsList) {
            recommendationsList.innerHTML = '';
            const recommendations = this.getRecommendations(result);
            recommendations.forEach(recommendation => {
                const li = document.createElement('li');
                li.textContent = recommendation;
                recommendationsList.appendChild(li);
            });
        }

        // Show results
        results.classList.remove('hidden');
    }

    getRecommendations(result) {
        const recommendations = [];

        if (result.is_phishing) {
            recommendations.push('⚠️ Do not enter personal information on this website');
            recommendations.push('🚫 Avoid downloading files from this site');
            recommendations.push('📞 Contact the organization directly if you received a link via email');
        } else {
            recommendations.push('✅ Website appears to be legitimate');
            recommendations.push('🔍 Always verify URLs before entering sensitive information');
        }

        if (result.risk_score > 20) {
            recommendations.push('🛡️ Use additional security tools for verification');
            recommendations.push('📧 Be cautious if you arrived here via email link');
        }

        // Add ML-specific recommendations if available
        if (result.ml_probability && result.ml_probability > 0.5) {
            recommendations.push('🤖 Machine learning model detected suspicious patterns');
        }

        recommendations.push('🔒 Always look for HTTPS and valid certificates');
        recommendations.push('💡 When in doubt, navigate to the site directly');

        return recommendations;
    }

    showError(message) {
        const results = document.getElementById('results');
        if (results) {
            results.innerHTML = `
                <div class="error-message">
                    <h3>❌ Error</h3>
                    <p>${message}</p>
                    <div class="error-details">
                        <p><strong>Troubleshooting:</strong></p>
                        <ul>
                            <li>Make sure the Flask server is running on port 5000</li>
                            <li>Check if the URL format is correct (include http:// or https://)</li>
                            <li>Try refreshing the page and checking again</li>
                        </ul>
                    </div>
                </div>
            `;
            results.classList.remove('hidden');
        }
    }
}

// FIXED: Add error handling for initialization
document.addEventListener('DOMContentLoaded', () => {
    try {
        new PhishingDetector();
        console.log('✅ Phishing Detector initialized successfully');
    } catch (error) {
        console.error('❌ Error initializing Phishing Detector:', error);
    }
});

// FIXED: Add global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

// FIXED: Add unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});