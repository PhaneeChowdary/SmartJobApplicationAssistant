from turtle import st
import plotly.graph_objects as go
import plotly.express as px
import json

def create_radar_chart(skills_data):
    """Create a radar chart for skills matching."""
    fig = go.Figure()
    
    # Add traces for actual skills
    fig.add_trace(go.Scatterpolar(
        r=[skill['score'] for skill in skills_data],
        theta=[skill['name'] for skill in skills_data],
        fill='toself',
        name='Your Skills',
        line_color='#3b82f6',
        fillcolor='rgba(59, 130, 246, 0.5)'
    ))
    
    # Add traces for required skills (100%)
    fig.add_trace(go.Scatterpolar(
        r=[100] * len(skills_data),
        theta=[skill['name'] for skill in skills_data],
        fill='toself',
        name='Required Skills',
        line_color='#dc2626',
        fillcolor='rgba(220, 38, 38, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Skills Match Analysis"
    )
    
    return fig

def create_category_chart(category_data):
    """Create a bar chart for category scores."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[cat['category'] for cat in category_data],
        y=[cat['score'] for cat in category_data],
        marker_color='#3b82f6',
        name='Match Percentage'
    ))
    
    fig.update_layout(
        title="Category Match Analysis",
        xaxis_title="Category",
        yaxis_title="Match Percentage (%)",
        yaxis=dict(range=[0, 100]),
        xaxis_tickangle=-45,
        height=400
    )
    
    return fig

def create_gauge_chart(score):
    """Create a gauge chart for overall match score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Match Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [50, 75], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [75, 100], 'color': "rgba(34, 197, 94, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig

def display_match_analysis(analysis_text, scores_data):
    """Display the complete analysis with visualizations."""
    try:
        # Display overall score gauge
        st.plotly_chart(create_gauge_chart(scores_data['overallScore']), use_container_width=True)
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Display skills radar chart
            st.plotly_chart(create_radar_chart(scores_data['skillsMatch']), use_container_width=True)
        
        with col2:
            # Display category bar chart
            st.plotly_chart(create_category_chart(scores_data['categoryScores']), use_container_width=True)
        
        # Display detailed analysis
        st.markdown("## Detailed Analysis")
        st.markdown(analysis_text)
        
    except Exception as e:
        st.error(f"Error displaying analysis: {str(e)}")