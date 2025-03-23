from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
import json
from datetime import datetime
from agentmethods import start_interview_process

app = Flask(__name__)
app.secret_key = 'aicruitersecretkey'

os.makedirs('uploads', exist_ok=True)
os.makedirs('data', exist_ok=True)

AGENT_DETAILS_FILE = os.path.join('data', 'agentdetails.json')
DETAILS_FILE = os.path.join('data', 'details.json')

def get_agents():
    """
    Get agents as a list (for UI display)
    """
    try:
        print(f"Loading agents from: {AGENT_DETAILS_FILE}")
        if os.path.exists(AGENT_DETAILS_FILE) and os.path.getsize(AGENT_DETAILS_FILE) > 0:
            with open(AGENT_DETAILS_FILE, 'r') as f:
                agents_dict = json.load(f)
                print(f"Loaded agents dict: {agents_dict}")
                # Convert dict to list for the UI
                agents_list = list(agents_dict.values())
                print(f"Converted to list: {agents_list}")
                return agents_list
        else:
            print("No agents file exists or it's empty, creating defaults")
            # If no agents exist yet, create defaults and save to agentdetails.json
            default_agents = _get_default_agents()
            save_agents(default_agents)
            return default_agents
    except Exception as e:
        print(f"Error loading agents: {str(e)}")
        default_agents = _get_default_agents()
        save_agents(default_agents)
        return default_agents

def _get_default_agents():
    return [
        {
            "id": "software-engineer",
            "title": "Software Engineer",
            "category": "Technology",
            "values": "Innovation, collaboration, and excellence are our core values.",
            "description": "We're looking for a Software Engineer to develop high-quality applications.",
            "personality": {
                "type": "professional",
                "description": ""
            },
            "criteria": [
                "5+ years of software development experience",
                "Proficiency in JavaScript and React",
                "Experience with cloud platforms (AWS, Azure, GCP)"
            ]
        },
        {
            "id": "product-manager",
            "title": "Product Manager",
            "category": "Product",
            "values": "User-focused, data-driven decision making",
            "description": "We need a Product Manager who can translate business goals into product features.",
            "personality": {
                "type": "professional",
                "description": ""
            },
            "criteria": []
        }
    ]

def save_agents(agents_list):
    """
    Save a list of agents to the agentdetails.json file
    """
    try:
        # Convert list to dictionary with id as key
        agents_dict = {}
        for agent in agents_list:
            agents_dict[agent['id']] = agent
            
        with open(AGENT_DETAILS_FILE, 'w') as f:
            json.dump(agents_dict, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving agents: {str(e)}")
        return False

def get_candidate_details():
    if os.path.exists(DETAILS_FILE):
        try:
            with open(DETAILS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"candidates": []}
    else:
        return {"candidates": []}

def save_candidate_details(details):
    with open(DETAILS_FILE, 'w') as f:
        json.dump(details, f, indent=2)

def save_agent_details(agent_data):
    """Save agent details to agentdetails.json file"""
    try:
        if os.path.exists(AGENT_DETAILS_FILE) and os.path.getsize(AGENT_DETAILS_FILE) > 0:
            with open(AGENT_DETAILS_FILE, 'r') as f:
                agents_dict = json.load(f)
        else:
            agents_dict = {}
        agents_dict[agent_data['id']] = agent_data
        with open(AGENT_DETAILS_FILE, 'w') as f:
            json.dump(agents_dict, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving agent details: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            agent_id = request.form['agent']
            
            resume_filename = ''
            if 'resume' in request.files and request.files['resume'].filename:
                resume = request.files['resume']
                resume_filename = f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{resume.filename.split('.')[-1]}"
                resume.save(os.path.join('uploads', resume_filename))
            
            links = request.form.getlist('links[]')
            links = [link for link in links if link.strip()]
            
            agent_details = None
            if os.path.exists(AGENT_DETAILS_FILE):
                with open(AGENT_DETAILS_FILE, 'r') as f:
                    agents_dict = json.load(f)
                    if agent_id in agents_dict:
                        agent_details = agents_dict[agent_id]
            
            details = {
                'name': name,
                'email': email,
                'resume': resume_filename,
                'links': links,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'agent': agent_details
            }
            save_candidate_details(details)
            
            success, message = start_interview_process()
            
            if success:
                flash('Interview request sent successfully! The candidate will receive an email with the interview link.', 'success')
            else:
                flash(f'Interview setup failed: {message}', 'danger')
                
            return redirect(url_for('page1'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    
    agents = get_agents()
    print(f"Agents being passed to template: {agents}")
    if not agents:
        print("WARNING: No agents available to display in the dropdown!")
    return render_template('page1.html', agents=agents)

@app.route('/page2')
def page2():
    agents = get_agents()
    return render_template('page2.html', agents=agents)

@app.route('/api/save-agent', methods=['POST'])
def save_agent():
    try:
        agent_data = request.json
        # Ensure we're using company_values instead of values
        if 'values' in agent_data:
            agent_data['company_values'] = agent_data.pop('values')
        save_agent_details(agent_data)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/delete-agent/<agent_id>', methods=['DELETE'])
def delete_agent(agent_id):
    try:
        if os.path.exists(AGENT_DETAILS_FILE):
            with open(AGENT_DETAILS_FILE, 'r') as f:
                agents_dict = json.load(f)
            if agent_id in agents_dict:
                del agents_dict[agent_id]
                with open(AGENT_DETAILS_FILE, 'w') as f:
                    json.dump(agents_dict, f, indent=4)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, threaded=False, use_reloader=False)
