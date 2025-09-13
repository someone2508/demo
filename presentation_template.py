#!/usr/bin/env python3
"""
Generate presentation slides for the Player Analytics Assignment
This creates an HTML presentation with all key insights and visualizations
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_presentation():
    """Create an HTML presentation with all key insights"""
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Player Analytics - Interview Presentation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        
        .presentation-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .slide {
            background: white;
            border-radius: 15px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            page-break-after: always;
            min-height: 600px;
        }
        
        .slide-header {
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        h2 {
            color: #764ba2;
            font-size: 2em;
            margin-bottom: 20px;
        }
        
        h3 {
            color: #667eea;
            font-size: 1.5em;
            margin: 20px 0 10px 0;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.2em;
            font-style: italic;
        }
        
        .key-metric {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            display: inline-block;
            min-width: 200px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            display: block;
        }
        
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 5px;
        }
        
        .insight-box {
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        .recommendation {
            background: #e8f5e9;
            border-left: 5px solid #4caf50;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        .warning {
            background: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }
        
        .table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        
        .table tr:hover {
            background: #f5f5f5;
        }
        
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .bullet-list {
            margin: 20px 0;
            padding-left: 20px;
        }
        
        .bullet-list li {
            margin: 10px 0;
            font-size: 1.1em;
            line-height: 1.6;
        }
        
        .highlight {
            background: yellow;
            padding: 2px 5px;
            border-radius: 3px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: white;
            margin-top: 40px;
        }
        
        @media print {
            .slide {
                page-break-after: always;
            }
        }
    </style>
</head>
<body>
    <div class="presentation-container">
        
        <!-- Slide 1: Title Slide -->
        <div class="slide">
            <div class="slide-header">
                <h1>Player Analytics Assignment</h1>
                <p class="subtitle">Comprehensive Analysis of Player Behavior & Value Segmentation</p>
            </div>
            <div style="margin-top: 50px;">
                <h3>Presentation Outline</h3>
                <ul class="bullet-list">
                    <li>Executive Summary & Key Findings</li>
                    <li>Funnel & Conversion Analysis</li>
                    <li>Retention & Engagement Patterns</li>
                    <li>Player Segmentation & Value Distribution</li>
                    <li>Strategic Recommendations</li>
                    <li>Technical Implementation</li>
                </ul>
            </div>
            <div style="margin-top: 50px; text-align: center;">
                <p style="color: #666;">Prepared using Python | pandas | scikit-learn | plotly</p>
            </div>
        </div>
        
        <!-- Slide 2: Executive Summary -->
        <div class="slide">
            <div class="slide-header">
                <h2>Executive Summary</h2>
                <p class="subtitle">Key Performance Indicators</p>
            </div>
            
            <div class="grid">
                <div class="key-metric">
                    <span class="metric-value">36.7%</span>
                    <span class="metric-label">Registration → Deposit</span>
                </div>
                <div class="key-metric">
                    <span class="metric-value">41.7%</span>
                    <span class="metric-label">Registration → First Bet</span>
                </div>
                <div class="key-metric">
                    <span class="metric-value">29.0%</span>
                    <span class="metric-label">30-Day Retention</span>
                </div>
                <div class="key-metric">
                    <span class="metric-value">42.7%</span>
                    <span class="metric-label">Top 10% Revenue Share</span>
                </div>
            </div>
            
            <div class="insight-box">
                <h3>🎯 Primary Finding</h3>
                <p>The player funnel shows significant drop-off at the first deposit stage (63.3% loss), 
                representing the primary conversion bottleneck. However, once players deposit, 
                engagement is strong with 85.9% placing their first bet within 24 hours.</p>
            </div>
            
            <div class="recommendation">
                <h3>💡 Key Opportunity</h3>
                <p>Optimizing acquisition channels could improve conversion by 15-20%. 
                Paid Search and Affiliate channels show 46% deposit rates vs 28% for Direct traffic.</p>
            </div>
        </div>
        
        <!-- Slide 3: Funnel Analysis -->
        <div class="slide">
            <div class="slide-header">
                <h2>Funnel & Conversion Analysis</h2>
                <p class="subtitle">Player Journey from Registration to Activity</p>
            </div>
            
            <table class="table">
                <thead>
                    <tr>
                        <th>Stage</th>
                        <th>Players</th>
                        <th>Conversion Rate</th>
                        <th>Drop-off</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Registrations</td>
                        <td>10,000</td>
                        <td>100%</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>First Deposit</td>
                        <td>3,667</td>
                        <td>36.7%</td>
                        <td class="highlight">63.3%</td>
                    </tr>
                    <tr>
                        <td>First Bet</td>
                        <td>4,168</td>
                        <td>41.7%</td>
                        <td>58.3%</td>
                    </tr>
                    <tr>
                        <td>Active (30 days)</td>
                        <td>2,896</td>
                        <td>29.0%</td>
                        <td>71.0%</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="insight-box">
                <h3>Channel Performance</h3>
                <ul class="bullet-list">
                    <li><strong>Best:</strong> Affiliate (46.2% deposit rate)</li>
                    <li><strong>Good:</strong> Paid Search (45.3% deposit rate)</li>
                    <li><strong>Poor:</strong> Direct (28.0% deposit rate)</li>
                </ul>
            </div>
            
            <div class="warning">
                <h3>⚠️ Critical Bottleneck</h3>
                <p>Registration → First Deposit shows the highest drop-off (63.3%). 
                This represents the most significant opportunity for improvement.</p>
            </div>
        </div>
        
        <!-- Slide 4: Retention Analysis -->
        <div class="slide">
            <div class="slide-header">
                <h2>Retention & Engagement Analysis</h2>
                <p class="subtitle">Player Activity Patterns in First 30 Days</p>
            </div>
            
            <h3>Activity Cohort Distribution</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Activity Level</th>
                        <th>Players</th>
                        <th>% of Active</th>
                        <th>Deposit Contribution</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1-2 days</td>
                        <td>259</td>
                        <td>8.9%</td>
                        <td>0.0%</td>
                    </tr>
                    <tr>
                        <td>3-5 days</td>
                        <td>1,005</td>
                        <td>34.7%</td>
                        <td>0.3%</td>
                    </tr>
                    <tr>
                        <td>6-10 days</td>
                        <td>816</td>
                        <td>28.2%</td>
                        <td>31.3%</td>
                    </tr>
                    <tr>
                        <td class="highlight">11-30 days</td>
                        <td class="highlight">816</td>
                        <td class="highlight">28.2%</td>
                        <td class="highlight">68.4%</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="grid">
                <div class="insight-box">
                    <h3>⏱️ Speed to Action</h3>
                    <p><strong>8.3 hours</strong> median time from deposit to first bet</p>
                    <p><strong>85.9%</strong> bet within same day of deposit</p>
                </div>
                <div class="recommendation">
                    <h3>📈 Growth Opportunity</h3>
                    <p>Players active 11-30 days generate 68.4% of deposits despite being only 28.2% of active players</p>
                </div>
            </div>
        </div>
        
        <!-- Slide 5: Player Segmentation -->
        <div class="slide">
            <div class="slide-header">
                <h2>Player Segmentation Analysis</h2>
                <p class="subtitle">Value Distribution & Customer Segments</p>
            </div>
            
            <h3>Value Concentration (Pareto Analysis)</h3>
            <div class="grid">
                <div class="key-metric">
                    <span class="metric-value">1%</span>
                    <span class="metric-label">contribute 14.5% of deposits</span>
                </div>
                <div class="key-metric">
                    <span class="metric-value">5%</span>
                    <span class="metric-label">contribute 30.0% of deposits</span>
                </div>
                <div class="key-metric">
                    <span class="metric-value">10%</span>
                    <span class="metric-label">contribute 42.7% of deposits</span>
                </div>
            </div>
            
            <h3>Natural Player Segments (K-Means Clustering)</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Segment</th>
                        <th>Players</th>
                        <th>Avg Deposit</th>
                        <th>Range</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Micro Stakes</td>
                        <td>1,162</td>
                        <td>$13.35</td>
                        <td>$2-19</td>
                    </tr>
                    <tr>
                        <td>Low Stakes</td>
                        <td>1,330</td>
                        <td>$27.75</td>
                        <td>$19-44</td>
                    </tr>
                    <tr>
                        <td>Mid Stakes</td>
                        <td>782</td>
                        <td>$74.51</td>
                        <td>$44-115</td>
                    </tr>
                    <tr>
                        <td class="highlight">High Rollers</td>
                        <td class="highlight">393</td>
                        <td class="highlight">$223.90</td>
                        <td class="highlight">$115+</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="insight-box">
                <h3>💰 Revenue Insights</h3>
                <p>High Rollers represent only 10.7% of depositors but generate disproportionate value. 
                The minimum deposit to qualify for top 10% is $118.81.</p>
            </div>
        </div>
        
        <!-- Slide 6: Strategic Recommendations -->
        <div class="slide">
            <div class="slide-header">
                <h2>Strategic Recommendations</h2>
                <p class="subtitle">Prioritized Action Plan</p>
            </div>
            
            <h3>🚀 Immediate Actions (0-30 days)</h3>
            <div class="recommendation">
                <h4>1. Channel Optimization</h4>
                <ul class="bullet-list">
                    <li>Reallocate budget to Affiliate and Paid Search channels</li>
                    <li>Expected impact: <strong>15-20% conversion improvement</strong></li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h4>2. Fast-Track Onboarding</h4>
                <ul class="bullet-list">
                    <li>Implement 24-48 hour time-limited welcome bonuses</li>
                    <li>Target: <strong>Reduce time-to-deposit by 30%</strong></li>
                </ul>
            </div>
            
            <h3>📊 Medium-term Initiatives (30-90 days)</h3>
            <div class="insight-box">
                <h4>3. VIP Program Development</h4>
                <ul class="bullet-list">
                    <li>Auto-identify deposits >$250 for VIP treatment</li>
                    <li>Assign dedicated support for top 10% depositors</li>
                </ul>
            </div>
            
            <div class="insight-box">
                <h4>4. Retention Intervention</h4>
                <ul class="bullet-list">
                    <li>Automated campaigns at Day 3 and Day 5</li>
                    <li>Personalized offers based on deposit tier</li>
                </ul>
            </div>
            
            <h3>🎯 Long-term Strategy (90+ days)</h3>
            <div class="warning">
                <h4>5. Predictive Analytics</h4>
                <ul class="bullet-list">
                    <li>Build churn prediction models</li>
                    <li>Develop LTV predictions for resource allocation</li>
                </ul>
            </div>
        </div>
        
        <!-- Slide 7: Technical Implementation -->
        <div class="slide">
            <div class="slide-header">
                <h2>Technical Implementation</h2>
                <p class="subtitle">Methodology & Tools Used</p>
            </div>
            
            <h3>🔧 Analysis Methodology</h3>
            <ul class="bullet-list">
                <li><strong>Data Processing:</strong> Python with pandas for ETL and data manipulation</li>
                <li><strong>Statistical Analysis:</strong> NumPy and SciPy for calculations</li>
                <li><strong>Segmentation:</strong> K-means clustering (scikit-learn) on log-transformed deposits</li>
                <li><strong>Visualization:</strong> Plotly for interactive dashboards</li>
            </ul>
            
            <h3>📈 Key Metrics Tracked</h3>
            <div class="grid">
                <div class="insight-box">
                    <h4>Funnel Metrics</h4>
                    <ul>
                        <li>Stage conversion rates</li>
                        <li>Channel performance</li>
                        <li>Drop-off analysis</li>
                    </ul>
                </div>
                <div class="insight-box">
                    <h4>Engagement Metrics</h4>
                    <ul>
                        <li>Time to first action</li>
                        <li>30-day activity levels</li>
                        <li>Cohort retention</li>
                    </ul>
                </div>
                <div class="insight-box">
                    <h4>Value Metrics</h4>
                    <ul>
                        <li>Deposit distribution</li>
                        <li>Gini coefficient</li>
                        <li>Segment profitability</li>
                    </ul>
                </div>
            </div>
            
            <h3>🔄 Next Steps for Production</h3>
            <ol class="bullet-list">
                <li>Implement real-time data pipeline</li>
                <li>Set up automated reporting dashboards</li>
                <li>Create A/B testing framework</li>
                <li>Deploy predictive models to production</li>
            </ol>
        </div>
        
        <!-- Slide 8: Q&A -->
        <div class="slide">
            <div class="slide-header">
                <h2>Questions & Discussion</h2>
                <p class="subtitle">Thank you for your time</p>
            </div>
            
            <div style="margin-top: 80px; text-align: center;">
                <h3>Key Takeaways</h3>
                <div class="grid" style="margin-top: 40px;">
                    <div class="insight-box">
                        <h4>✅ Clear Opportunities</h4>
                        <p>Channel optimization and deposit conversion represent immediate wins</p>
                    </div>
                    <div class="insight-box">
                        <h4>✅ Data-Driven Approach</h4>
                        <p>All recommendations backed by quantitative analysis</p>
                    </div>
                    <div class="insight-box">
                        <h4>✅ Scalable Solution</h4>
                        <p>Framework ready for production deployment</p>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 80px; text-align: center;">
                <p style="color: #666; font-size: 1.2em;">
                    📧 Ready to discuss implementation details<br>
                    📊 Additional analysis available upon request
                </p>
            </div>
        </div>
        
    </div>
    
    <div class="footer">
        <p>Player Analytics Assignment | Comprehensive Analysis Report</p>
    </div>
</body>
</html>
    """
    
    # Save the presentation
    with open('/workspace/presentation.html', 'w') as f:
        f.write(html_content)
    
    print("✅ Presentation created successfully!")
    print("📁 Open 'presentation.html' in your browser to view the slides")
    print("\n💡 Tips for presenting:")
    print("  • Use full-screen mode (F11) for best viewing")
    print("  • Each slide is designed to fit on one screen")
    print("  • Print to PDF for easy sharing")
    print("  • Customize colors and content as needed")

if __name__ == "__main__":
    create_presentation()