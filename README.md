# Player Analytics Assignment - Complete Solution

## 📋 Overview

This repository contains a comprehensive solution for the Player Analytics interview assignment. The analysis covers:

1. **Funnel & Conversion Analysis** - Player progression from registration to activity
2. **Retention & Engagement Analysis** - 30-day activity patterns and cohort behavior  
3. **Player Segmentation** - Value distribution and customer profiling

## 🚀 Quick Start

### Step 1: Download Your Data

1. Go to the Google Drive link: https://drive.google.com/drive/folders/1hgJPs06K4Q6wyFnYpBYvTVN8hYkJLW3e
2. Download all 5 CSV files
3. Create a folder: `/workspace/data/`
4. Place all CSV files in this folder

### Step 2: Run the Analysis

#### Option A: Use Sample Data (Already Complete)
```bash
python3 player_analytics_assignment.py
```
This runs the analysis on generated sample data to demonstrate capabilities.

#### Option B: Use Your Real Data
1. Edit `load_real_data.py` and update the filenames (lines 23-29) to match your CSV files
2. Run:
```bash
python3 load_real_data.py
```

### Step 3: View Results

The analysis generates several output files:

| File | Description |
|------|-------------|
| `player_analytics_dashboard.html` | Main interactive dashboard with all visualizations |
| `channel_heatmap.html` | Detailed channel performance analysis |
| `cohort_deposits.html` | Cohort retention and value analysis |
| `deposit_violin.html` | Deposit distribution visualization |
| `analysis_report.md` | Comprehensive written report |
| `presentation.html` | **Ready-to-present slides for your interview** |

## 📊 Generated Visualizations

### 1. Main Dashboard
- Player funnel visualization
- Conversion rates by channel
- Activity distribution (30 days)
- Deposit amount distribution
- Lorenz curve (concentration analysis)
- Player segments breakdown

### 2. Supporting Charts
- Channel performance heatmap
- Cohort retention analysis
- Deposit distribution patterns
- Time-to-action metrics

## 🎯 Key Insights (Sample Data)

### Funnel Performance
- **Primary bottleneck**: Registration → First Deposit (63.3% drop-off)
- **Best channel**: Affiliate (46.2% deposit rate)
- **Worst channel**: Direct (28.0% deposit rate)

### Value Concentration
- **Top 10% of players** contribute **42.7% of deposits**
- **Pareto effect** strongly present in player value distribution
- **4 natural segments** identified through clustering

### Engagement Patterns
- **85.9%** of depositors bet within **24 hours**
- **Median time** from deposit to bet: **8.3 hours**
- Players active **11-30 days** generate **68.4% of deposits**

## 💡 Strategic Recommendations

### Immediate (0-30 days)
1. **Channel Optimization** - Reallocate budget to high-performing channels
2. **Fast-Track Onboarding** - Time-limited welcome bonuses
3. **VIP Identification** - Flag high-value deposits immediately

### Medium-term (30-90 days)
1. **Retention Programs** - Automated campaigns at critical drop-off points
2. **Segmented Communication** - Tailored messaging by player segment
3. **A/B Testing Framework** - Optimize conversion at each funnel stage

### Long-term (90+ days)
1. **Predictive Modeling** - Churn and LTV prediction
2. **Product Development** - Features to boost early engagement
3. **Advanced Analytics** - Real-time dashboards and monitoring

## 🛠️ Technical Stack

- **Python 3.x** - Core programming language
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **scikit-learn** - Machine learning (K-means clustering)
- **Plotly** - Interactive visualizations
- **Seaborn/Matplotlib** - Statistical visualizations

## 📁 Project Structure

```
/workspace/
├── player_analytics_assignment.py  # Main analysis script (with sample data)
├── load_real_data.py               # Script to load and analyze your data
├── presentation_template.py         # Generate presentation slides
├── data/                           # Place your CSV files here
│   ├── registrations.csv
│   ├── deposits.csv
│   ├── bets.csv
│   ├── activity.csv
│   └── player_info.csv
└── output/                         # Generated reports and visualizations
    ├── player_analytics_dashboard.html
    ├── presentation.html
    └── analysis_report.md
```

## 🎤 Presentation Tips

1. **Open `presentation.html`** in your browser
2. Use **full-screen mode** (F11) for best viewing
3. Each slide is **self-contained** with key metrics
4. **Print to PDF** if needed for sharing
5. All numbers are **data-driven** and defensible

## ⚙️ Customization

### Adjusting for Your Data

The scripts automatically map common column names. If your data has different column names, update the mappings in `load_real_data.py`:

```python
# Example column mappings
column_mappings = {
    'user_id': 'player_id',
    'signup_date': 'registration_date',
    'first_deposit_amount': 'deposit_amount'
}
```

### Adding Custom Metrics

You can extend the analysis by adding custom metrics in the respective analysis functions:
- `funnel_conversion_analysis()` - Add conversion metrics
- `retention_engagement_analysis()` - Add engagement metrics
- `player_segmentation_analysis()` - Add segmentation criteria

## 📈 Expected Outcomes

After running the analysis, you'll be able to answer:

1. **Conversion rates** at each funnel stage
2. **Which acquisition channel** performs best
3. **Where the biggest drop-offs** occur
4. **Time gaps** between key actions
5. **Activity cohort** contribution to revenue
6. **Top 10% player** contribution
7. **Natural player segments** and their characteristics
8. **Clustering patterns** in deposit behavior

## 🤝 Interview Preparation

### Key Points to Emphasize

1. **Data-driven approach** - All recommendations backed by analysis
2. **Technical proficiency** - Python, SQL, statistical analysis
3. **Business understanding** - Focus on actionable insights
4. **Presentation skills** - Clear, structured findings
5. **Problem-solving** - Identified bottlenecks and solutions

### Questions You Might Get

- "How did you handle missing data?" → Discuss validation and cleaning
- "Why K-means for segmentation?" → Explain choice of algorithm
- "How would you implement this in production?" → Discuss data pipelines
- "What additional analyses would you do?" → Mention A/B testing, predictive modeling

## 🚨 Important Notes

1. **Data Privacy**: Ensure you have permission to use the provided data
2. **Validation**: Always validate results with domain knowledge
3. **Assumptions**: Document any assumptions made in the analysis
4. **Scalability**: Consider how solution would scale with more data

## 📞 Support

If you encounter any issues:

1. Check that all required packages are installed
2. Verify data file formats (CSV with headers)
3. Ensure column names are correctly mapped
4. Review error messages for specific issues

## ✅ Checklist Before Submission

- [ ] Analysis runs without errors
- [ ] All visualizations generated
- [ ] Report clearly explains findings
- [ ] Recommendations are actionable
- [ ] Presentation is polished
- [ ] Code is well-commented
- [ ] README is complete

## 🎉 Good Luck!

This comprehensive analysis should demonstrate:
- Technical capabilities (Python, data analysis, ML)
- Analytical reasoning (identifying patterns and insights)
- Business acumen (actionable recommendations)
- Presentation skills (clear, structured communication)

Remember to:
- **Customize** the analysis for your specific data
- **Practice** presenting the findings
- **Prepare** for follow-up questions
- **Show enthusiasm** for the role and company

---

*Built with Python | pandas | scikit-learn | plotly*