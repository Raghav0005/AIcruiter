from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
import json
from datetime import datetime
from agentmethods import start_interview_process

app = Flask(__name__)
app.secret_key = 'aicruitersecretkey'

os.makedirs('uploads', exist_ok=True)
os.makedirs('data', exist_ok=True)

AGENTS_FILE = 'data/agents.json'
DETAILS_FILE = 'details.json'

def get_agents():
    if os.path.exists(AGENTS_FILE):
        try:
            with open(AGENTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return _get_default_agents()
    else:
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            
            resume_filename = ''
            if 'resume' in request.files and request.files['resume'].filename:
                resume = request.files['resume']
                resume_filename = f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{resume.filename.split('.')[-1]}"
                resume.save(os.path.join('uploads', resume_filename))
            
            links = request.form.getlist('links[]')
            links = [link for link in links if link.strip()]
            
            details = {
                'name': name,
                'email': email,
                'resume': resume_filename,
                'links': links,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
    
    return render_template('page1.html')

@app.route('/page2')
def page2():
    agents = get_agents()
    return render_template('page2.html', agents=agents)

@app.route('/api/save-agent', methods=['POST'])
def save_agent():
    try:
        agent_data = request.json
        agents = get_agents()
        
        for i, agent in enumerate(agents):
            if agent['id'] == agent_data['id']:
                agents[i] = agent_data
                break
        else:
            agents.append(agent_data)
        
        save_agents(agents)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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
    # Run Flask in single-threaded mode and disable the reloader
    app.run(debug=True, threaded=False, use_reloader=False)
