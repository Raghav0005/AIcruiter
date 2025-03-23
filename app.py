from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
import json

app = Flask(__name__)
app.secret_key = 'aicruitersecretkey'  # Required for flash messages

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('data', exist_ok=True)

# File to store agents data
AGENTS_FILE = 'data/agents.json'

# Initialize or load agents
def get_agents():
    if os.path.exists(AGENTS_FILE):
        try:
            with open(AGENTS_FILE, 'r') as f:
                return json.load(f)
        except:
            # Return default if file is corrupted
            return _get_default_agents()
    else:
        # Create default agents file
        default_agents = _get_default_agents()
        save_agents(default_agents)
        return default_agents

def _get_default_agents():
    # Default agents if no file exists
    return [
        {
            "id": "software-engineer",
            "title": "Software Engineer",
            "category": "Technology",
            "values": "Innovation, collaboration, and excellence are our core values.",
            "description": "We're looking for a Software Engineer to develop high-quality applications."
        },
        {
            "id": "product-manager",
            "title": "Product Manager",
            "category": "Product",
            "values": "User-focused, data-driven decision making",
            "description": "We need a Product Manager who can translate business goals into product features."
        }
    ]

def save_agents(agents):
    with open(AGENTS_FILE, 'w') as f:
        json.dump(agents, f, indent=2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    # Get agents for the dropdown
    agents = get_agents()
    
    if request.method == 'POST':
        # Get form data
        candidate_name = request.form.get('name', '')
        uploaded_file = request.files.get('resume')
        agent_id = request.form.get('agent', '')
        links = request.form.getlist('links[]')
        
        # Process resume upload
        if uploaded_file and uploaded_file.filename:
            # Create a safe filename
            filename = os.path.join('uploads', f"{candidate_name.replace(' ', '_')}_resume{os.path.splitext(uploaded_file.filename)[1]}")
            uploaded_file.save(filename)
            flash(f'Application for {candidate_name} submitted successfully!', 'success')
            return redirect(url_for('page1'))
        else:
            flash('Please upload a resume file', 'danger')
    
    return render_template('page1.html', agents=agents)

@app.route('/page2')
def page2():
    # Pass the agents to page2 as well so they can be edited
    agents = get_agents()
    return render_template('page2.html', agents=agents)

# API endpoint to save agents from page2
@app.route('/api/save-agent', methods=['POST'])
def save_agent():
    try:
        agent_data = request.json
        agents = get_agents()
        
        # Check if agent exists to update or add new
        for i, agent in enumerate(agents):
            if agent['id'] == agent_data['id']:
                agents[i] = agent_data
                break
        else:
            # Agent doesn't exist, add it
            agents.append(agent_data)
        
        save_agents(agents)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# API endpoint to delete an agent
@app.route('/api/delete-agent/<agent_id>', methods=['DELETE'])
def delete_agent(agent_id):
    try:
        agents = get_agents()
        agents = [agent for agent in agents if agent['id'] != agent_id]
        save_agents(agents)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
