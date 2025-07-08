import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths for models and adapters
MODEL_NAME_QWEN = "/root/Qwen2.5-7B-Instruct"
LORA_ADAPTER_QWEN = "/root/autodl-tmp/sft/qwen7/lora/sft/"
MODEL_NAME_LLAMA = "/root/autodl-tmp/Llama-3.2-3B-Instruct"
LORA_ADAPTER_LLAMA = "/root/autodl-tmp/sft/llama3b/lora/sft/"

# Globals for model/tokenizer instances
model_qwen = None
tokenizer_qwen = None
model_llama = None
tokenizer_llama = None
MODELS_LOADED = False

# Few-shot examples for salary extraction
shots = """example1：
Job Title: Financial Account - Call Center Agent - Up to 34k
Job Description: Job Opening Financial Account - Call Center Agent - Up to 34k Job Industry Telecommunications Job Type Full-Time Experience Level Entry Level Date Posted 2022-10-27 Job Location Pasig BlvdPasig1000NCRPhilippines Company Information Sapient
 
 Pasig Blvd 
 Cebu, Central Visayas 
 6019 
 Sapient is Philippine-based BPO that provides a range of outsourcing services from consulting services, IT-enabled services, and call center services primarily catering small and medium based enterprises. Job Description Job Responsibilities: Answers phone calls and provides important information/ assistance to clients Checks mail, fax and internet mail to provide customer assistance Communicates with customer on the phone or using written correspondence to take care of concerns Answer participant questions, , as well as talk to participants to achieve full understanding of what critical information are being asked. Job Qualifications What are we looking for? Open to candidates who completed college Open to High School and Senior High School Graduates with BPO experience Excellent to above average English communication skills BPO experience of at least 6 months or have work experience Can do onsite work With in 25km to 35 km Compensation 17500 Compensation Range ₱15,000 - ₱20,000
Location: PH
y_true: 17500-17500-PHP-MONTHLY

example2：
Job Title: Aspiring Call Center Agents - Work from Home - Must be residing in Davao
Job Description: Job Opening Aspiring Call Center Agents - Work from Home - Must be residing in Davao Job Industry Telecommunications Job Type Full-Time Experience Level Entry Level Date Posted 2023-08-09 Job Location - Davao 8000 Davao del Sur Philippines Company Information Neksjob Corporation - We are driven by the innate desire to bring about change by encouraging out of the box solutions to well-worn path challenges at a cost-effective rate. We aim to bridge the gap between countries and cultures, distance and time zones, to bring the world closer through the help of emerging technology. Job Description Responsibilities: Answering incoming calls from customers Sorting out customers’ inquiries or requests Ensuring that customers’ requests are managed in an appropriate and timely manner Developing, organizing, and maintaining accurate files Delivering a high caliber of service in a friendly, confident, and informed manner Job Qualifications Qualifications: • Must be 18 years of age and above • At least high school graduate with diploma/certificate • Willing to work full-time and in shifting schedule (No Part-time) • Good to excellent English communication skills • Computer literate and with good web navigation skills • Available to start ASAP Work At Home Requirements: • Minimum upload speed of 5 MBPS • Minimum download speed of 10 MBPS • Wired connection from modem/router to PC • Conducive workspace away from distractions • Highly stable internet connection with no packet loss Compensation 16000 Compensation Range ₱15,000 - ₱20,000
Location: PH
y_true: 16000-16000-PHP-MONTHLY

example3：
Job Title: Production Staff Required - Afternoon & Night-shift
Job Description: Original Foods Baking Co. is one of New Zealand’s favourite wholesale bakeries.  We have been perfecting the art of good baking for over 30 years, and our products are dangerously good! Proudly New Zealand owned and operated, our donuts, cakes, brownies, muffins, slices, cookie pies and more are sold throughout New Zealand in supermarkets and wholesalers under the Original Foods Baking Co., Goofy and Bite Me brands. We are recruiting for experienced, strong, capable and motivated production staff to work in some of our more physically demanding roles. These roles will suit persons with the following attributes; Experience in a production environment Experience in mundane, repetitive roles Strong and physically fit Reliable, trustworthy, dependable Strong numeracy skills Team-player Please indicate which shift is your preference on application. Days & Hours of Work Monday - Friday Night shift 10.00pm - 6.30am Afternoon shift 2:00pm - 10:30pm You must be double vaccinated and have the right to live and work in New Zealand to apply for this role. Applications will be reviewed as they are received so apply NOW!
Location: NZ
y_true: 0-0-None-None

example4：
Job Title: Payer Analyst
Job Description: The Payer Analyst individual is assigned to the Revenue Cycle Management Product Management Content Intelligence team. The individual finds and analyzes health plans and payor websites for billing related information and enters selected information using a standardized data collection system and is responsible for setting up the website tracking and maintenance of the system that auto-tags relevant information contained in the content.  The individual assures, for each organization targeted that all are found and any relevant URLs and content are  are collected. Primary Duties/Responsibility 1:  Payer Content Research Researching Payer websites for content related to Healthcare Revenue Cycle Management and Billing. (Claims, Authorizations, Remittance Advice) Sign up for relevant email list-serves when available. Take the lead to proactively identify, track and distribute notifications to appropriate audiences to ensure issues are addressed before there is any impact to our customers. Identify new trends that impact the healthcare industry for potential in-scope expansion. Primary Duties/Responsibility 2:  Website Tracking & Maintenance Setup website tracking using “point & click” tools allowing for changes of websites to be identified. Tracking maintenance – resolve any tracking errors that may arise.  Monitor all tracking to ensure the latest versions of URLs/PDFs are input on a routine basis Primary Duties/Responsibility 3:  NLP Tagging Setup & Maintenance Setup the payer information in a Natural Language Processing system that allows for tagging of internal identifiers and content based subscription topics. Maintain payer set up by adding additional alternative labels as needed, i.e. when new tracking is added. Must be able to analyze overall system set-up concept to successfully resolve over/under tagging issues as they arise Primary Duties/Responsibility 4:  Curate Internal Notification & Bulletins Monitor incoming subscriptions and change tracking of payer websites to ensure NLP and Subscription topics are being properly associated with the information. Modify/split content into individual articles building internal “payer notices” to internal subscribers (There’s a lot of reading, copy & pasting.) Must maintain high standard in out-going content to Audiences with proper grammar, spelling and layout of the information. Maintain turn-around time performance standards to ensure timely release of information. Primary Duties/Responsibility 5:  Platform Migration Assist the team in migrating and testing above business processes to a new platform and start performing processes in new system. The ability to take on projects while maintaining current work-load; verbalizing the need for assistance when/if needed. The Payer Content Intelligence Analyst individual requires the ability to read and identify changes that need to be made to existing Claims Edits and Prior Auth policies as well as create internal bulletins based on their interpretation of the content. While the majority of the content is notifying internal teams of the website or subscription changes the written communication needs to be concise and accurate when documenting policy changes. Preferred Qualification/s: The ideal candidate is experienced in healthcare billing  with knowledge of managed care/payor/healthcare insurance companies who understands and has worked with medical policies, prior authorization, or utilization review.  Ability to comprehensively search the internet for payor content and accurate transcribe data to a prepared template is required. The preferred candidate will also have experience setting up NLP, Machine Learning, AI to assist with automatically tagging content. Minimum of 3 years of healthcare billing in an ambulatory or acute setting is required. Experience is other aspects of healthcare technology will also be considered.
Location: PH
y_true: 0-0-None-None

example5：
Job Title:   Solicitor, Restructuring (ID: 2100013K)
Job Description: The DLA Piper team operates across more than 40 countries, but we’re still locally connected. Our Restructuring & Insolvency team work on some of the most complex and interesting matters in the market. We partner with a diverse client base that includes debtors, lenders, government entities, trustees, shareholders, senior executives, as well as distressed debt and asset buyers and investors. We are currently looking for high performing restructuring solicitor with 5 + years’ experience to join the team based in our Melbourne office MAIN DUTIES AND RESPONSIBILITIES First class lawyers, experience and our unrivalled global coverage are just the beginning of what DLA Piper offers. By joining our team you will have the opportunity to work on a range of matters, advising clients on investigations, enforcement, litigation and asset recovery on a multijurisdictional basis. As part of our forward thinking team, you will share a client centric approach and take pride in delivering our clients sector focused advice across key expertise areas such as financial services, energy, resources & mining, retail, property and technology. ABOUT YOU We know talent is more than what's written on paper. It's the energy you bring and the way you work with your team. Your strong communication skills will enable you to develop and maintain high quality relationships with clients.  You will have a mature, confident approach and be highly motivated, thriving in a fast-paced practice. Your organisational skills and ability to manage your own workload, seeking input from team members where needed will support your success. You will receive first class on the job support and training, working with partners including  Lionel Meehan. You will also have access to our DLA Piper Career Academies, an award-winning international development forum designed to cultivate high performance and support your career goals. If you're a high performing restructuring lawyer ready to take the next step in your career at a firm that values you, we want to hear from you. Apply now and be part of our future. ABOUT US DLA Piper is a global law firm with lawyers located in more than 40 countries throughout the Americas, Europe, the Middle East, Africa and Asia Pacific. Our global reach ensures that we can help businesses with their legal needs anywhere in the world. We strive to be the leading global business law firm by delivering quality, service excellence and value to our clients and offering practical and innovative legal solutions to help them succeed. Our clients range from multinational, Global 1000, and Fortune 500 enterprises to emerging companies developing industry-leading technologies, as well as government and public sector bodies. OUR VALUES In everything we do connected with our People, our Clients and our Communities, we live by these values: Be Supportive - we care about others, value diversity and act thoughtfully Be Collaborative - we give, we share and we join in Be Bold - we stand tall and challenge ourselves to think big Be Exceptional - we exceed standards and expectations DIVERSITY AND INCLUSION At DLA Piper, diversity and inclusion underpins how we live our values and everything we do.  We believe that everyone has a voice, and that everyone’s voice counts. We know that the rich diversity across our firm makes us stronger, more innovative and creative, which helps us to better serve our clients and communities. We are committed to providing an inclusive working environment and culture across our global firm, where everyone can bring their authentic self to work. Diversity of perspective, thought, background and culture combine to make us the leading global law firm; that’s why we actively seek to build balanced teams. We welcome the unique contribution that you will bring to our firm and actively encourage applications from all talented people – however your talent is packaged, whatever your background or circumstance and regardless of how you identify. We support anyone with a disability or long term health condition to ensure they have the opportunity to perform at their best. If you have not done so already, please let us know if you require any support so we can make the right adjustments and considerations should they be required. AGILE WORKING We know that people have responsibilities and interests outside of their career and that as a business, we all benefit from working flexibly. That's why we are open to discussing with candidates the different ways in which we are able to support requests for agile working arrangements. PRE-ENGAGEMENT SCREENING In the event that we make an offer to you, and where local legislation permits, we will conduct pre-engagement screening checks that may include but are not limited to your professional and academic qualifications, your eligibility to work in the relevant jurisdiction, any criminal records, your financial stability and references from previous employers.
Location: AUS
y_true: 0-0-None-None

example6：
Job Title: Sanrio Gift Gate 兼職店務員(馬鞍山)
Job Description: Sanrio Gift Gate 兼職店務員(馬鞍山) 工作內容: 負責精品店舖Sanrio Gift Gate的日常運作，提供優質客戶服務，銷售貨品及貨品陳列 要求: 具有關工作經驗優先, 有責任感, 積極主動, 熱誠, 具良好銷售技巧優先 工作時間: 上午10時 至 晚上10時, 每週工作3-5天 ，每更工作6-9小時 工作地點: 馬鞍山Sanrio Gift Gate分店 兼職時薪: $50 - $60 有意應徵者可 (1) 電郵個人履歷至recruitretail@danielco.com.hk 或 (2) 致電招聘熱線2149 8621 或   (3) 招聘whatsapp 6232 3687 或 https://forms.gle/7n1o8yFAfZTxuZkz5 填寫資料 <申請人所提供的資料將予保密及只作招聘用途>
Location: HK
y_true: 50-60-HKD-HOURLY

example7：
Job Title: Key Account Manager
Job Description: This role will contribute to expand our client Indonesia's customer base (B2B) and profit and loss statement by increasing the engagement with current customers and effectively bringing new customers on board. Client Details Our client is a global leader in packaging solutions. Description Contribute to the identification of products within the customer portfolio that demonstrate high market potential, competitiveness, and align with the client manufacturing capabilities. Participate in establishing tactical or strategic partnerships with customers to ensure the long-term supply of the products. Undertake customer research activities. Aid in the development of market insights and analyses for senior stakeholders. Assist in crafting business plans for new products and strategic partnerships. Collaborate with internal cross-functional teams to ensure the successful delivery of these products and fulfil client's commitments to its customers. Manage external stakeholders within the customer organisation in line with stakeholder mapping. Organise business planning events and periodic performance reviews (quarterly, semi-annually, annually) with customers. Facilitate the on boarding process for new customers and arrange visits to client's plants. Manage sales processes between client and customers. Develop monthly sales forecasts in conjunction with the customer. Support joint tactical marketing events with the marketing team or other departments. Profile A D3 degree holder Have minimum 3 years of experience and been at least once in sales or business development role Proficiency in English will be a plus Business Acumen Commercial Awareness: Capable of comprehending and grasping business concepts and strategies. Autonomous: Demonstrated track record of delivering results independently as an individual contributor. Analytic Proficiency : Competent in independently gathering and organising customer data for effective presentations. Previous involvement in customer and competitor research is a plus. Product Development Management of Product Life cycle : Previous experience in participating in the initial phases of product introduction to new customers or launching new products is advantageous. Analytic Thinking : Proficient in collecting customer insights, identifying opportunities, and offering actionable recommendations. Sales Process Effective Key Account Management : Previous involvement in managing the sales process from prospecting to achieving profitable outcomes. Soft Skills Interpersonal Adaptability : Capable of adjusting communication style and approach based on the audience. Skilled at building rapport with stakeholders at various levels. Stakeholder Engagement : Effective collaboration with cross-functional teams and establishing productive relationships with external stakeholders. Self-Motivated : Able to work independently with minimal supervision. Disciplined, proactive, and willing to go the extra mile, including working beyond standard office hours and travelling. Quick Learning Ability: Demonstrated willingness and eagerness to independently seek out information and learn about new businesses, products, and markets. Presentation Proficiency : Strong presentation skills and proficient in using the Microsoft Office suite. Job Offer Exciting career advancement opportunities abound within the company due to its remarkable growth potential. The opportunity to be part of a global company. A chance to significantly influence the commercial decisions of a highly successful enterprise and contribute to the development, execution, and oversight of systems from the outset. Competitive compensation package and bonus. To apply online please click the 'Apply' button below. For a confidential discussion about this role please contact Cheren Filus on +62 21 2958 8838.
Location: ID
y_true: 0-0-None-None

example8：
Job Title: Customer Service Agent With 1 month Call Center Experience
Job Description: Job Opening Customer Service Agent With 1 month Call Center Experience Job Industry Telecommunications Job Type Full-Time Experience Level Associate Date Posted 2023-08-15 Job Location -Baguio2600Baguio CityPhilippines Company Information Neksjob Corporation
 
 - 
 We are driven by the innate desire to bring about change by encouraging out of the box solutions to well-worn path 
challenges at a cost-effective rate. We aim to bridge the gap between countries and cultures, distance and time zones,
to bring the world closer through the help of emerging technology. Job Description Duties/Responsibilities:
Responsible for taking incoming calls or making outgoing calls for a business in a call center. These calls may be for a variety of situations, such as customer service, sales calls, product instructions, and billing inquiries. You may be responsible for taking orders, handling customer complaints, and answering questions from callers.
Why pick us?
Competitive Salary
Exciting Performance Bonuses & Account Specific Allowances
Career Advancement Opportunities
Promote Within the Company
Comprehensive Healthcare Benefits Job Qualifications What are we looking for?
Good to excellent communication skills 
At least 1 month Call Center Experience
Amenable to Work On-site Compensation 16000 Compensation Range ₱15,000 - ₱20,000 Inquire Apply for Job 
 Cancel
Location: PH
y_true: 16000-16000-PHP-MONTHLY

example9：
Job Title: ASAP - HR AND ADMIN ASSISTANT
Job Description: Job Opening ASAP - HR AND ADMIN ASSISTANT Job Industry Recreational Facilities Services Job Type Full-Time Experience Level Entry Level Date Posted 2023-10-25 Job Location Makati City Makati 1226 Metro Manila Philippines Company Information I-TECH DIGITAL PRODUCTIONS, INC Makati City Palanan, Makati City 1226 Interested applicants may email their resumes. Do not attach your resume here, just send it to claire2023.itechdigital@gmail.com Job Description Conducting interviews, recruiting, and vetting new staff. Arranging training sessions with all new hires and refresher workshops for existing employees. Assisting managers with staff requirements. Identifying and addressing employee requirements regarding performance issues, training, and career growth. Performing various administrative tasks and accurately processing paperwork. Counseling staff on HR policies, practices, and procedures. Job Qualifications *Bachelor’s degree in Human Resources. *Minimum 2 years of relevant experience in human resources. *Prepare HR documents, like employment contracts and new hire guides. *Answer employees queries about HR-related issues. *Assist payroll department by providing relevant employee information (e.g. leaves of absence, sick days and work schedules). *Ensures that policies are fully implemented and performance management and other such staff issues are addressed appropriately in line with the labour law and in a timely manner. *Excellent verbal and written communication skills. *Strong interpersonal skills and proven ability to handle diverse sources of information in a confidential, sensitive manner with due care, respect and discretion and absolute confidentiality *Full understanding of HR functions esp. IR process. *Additional training/certification in Payroll Management – may be advantageous but not required. *Willing to work in Makati City * Can start ASAP / Job types: FULL TIME. Compensation 17500 Compensation Range ₱15,000 - ₱20,000 Number of Job Opening 2 Highest Education Attainment College Graduate
Location: PH
y_true: 17500-17500-PHP-MONTHLY

example10：
Job Title: Brand Ambassador
Job Description: We have a super exciting opportunity in Queenstown to work as a brand ambassador for a new spirit company hitting our shores. For this campaign we are recruiting brand ambassadors, who confident, enjoy going up to strangers and get excited. Due to the nature of the client, all candidates will be interviewed. Reporting to our Manager, this role is all about sales & giving out merchandise! Providing outstanding customer service in-store Driving in-store sales Working closely within a team dynamic Working closely with sales reps Own Transport Can start immediately To be successful in the role, you will: Be a sales focused dynamo and love bar, retail sales Be positive, friendly and approachable Share our passion for spirits Be a genuine people person – you are confident and thrive on making new connections and comfortable being around all walks of life Reliable - you turn up to work and on time and have a reliable form of transport to and from work Well-presented Be able to work independently and confidently in a sole charge capacity A team player and enjoy collaborating with your team Willing to learn and to teach others Have a car Mileage and Travel will be reimbursed if over 20KMS Applicants must: Be 25 years or older Can commit to 20 hours per week across 6 weeks Must be able to work as an Independent Contractor within New Zealand Campaign kicks off 1st August Have previous retail sales experience and a proven track record and ability to generate sales. So, what are you waiting for? Join a team of passionate people who push boundaries and create memorable experiences. To apply for this role you need to have the right to work in New Zealand as an Independent Contractor. Job Type: Freelance Contract length: Pay: From $32.00 per hour Schedule: Rotating roster Expected Start Date: ASAP
Location: NZ
y_true: 32-32-NZD-HOURLY'"""

# Instruction templates
INSTRUCTION_TEMPLATES = {
    "salary": "You are a salary extraction assistant. Please extract the salary range from the job description in the format '100-200-AUS-MONTHLY'. If no salary is mentioned, return '0-0-None-None'.",
    "work arrangement": "You are a classification assistant. Extract the work arrangement from the job description: choose OnSite, Remote, or Hybrid.",
    "seniority": "You are a classification assistant. Extract the seniority level from the job description (e.g., junior, senior, entry level, lead, etc.)."
}

def load_models():
    global model_qwen, tokenizer_qwen, model_llama, tokenizer_llama, MODELS_LOADED
    if MODELS_LOADED:
        return

    # Load Qwen with LoRA
    tokenizer_qwen = AutoTokenizer.from_pretrained(MODEL_NAME_QWEN, trust_remote_code=True)
    base_qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_QWEN, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model_qwen = PeftModel.from_pretrained(base_qwen, LORA_ADAPTER_QWEN)

    # Load Llama with LoRA
    tokenizer_llama = AutoTokenizer.from_pretrained(MODEL_NAME_LLAMA, trust_remote_code=True)
    base_llama = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_LLAMA, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model_llama = PeftModel.from_pretrained(base_llama, LORA_ADAPTER_LLAMA)

    MODELS_LOADED = True


def get_model_response(model, tokenizer, messages, max_new_tokens=60):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()


def process_input(qtype, description):
    load_models()
    qtype = qtype.lower()

    if qtype == 'salary':
        # Salary extraction with few-shot examples
        messages = [
            {"role": "system", "content": INSTRUCTION_TEMPLATES['salary']},
            {"role": "user", "content": f"{shots}\n{description}"}
        ]
        return get_model_response(model_qwen, tokenizer_qwen, messages)

    elif qtype == 'work arrangement':
        messages = [
            {"role": "system", "content": INSTRUCTION_TEMPLATES['work arrangement']},
            {"role": "user", "content": description}
        ]
        return get_model_response(model_qwen, tokenizer_qwen, messages)

    elif qtype == 'seniority':
        messages = [
            {"role": "system", "content": INSTRUCTION_TEMPLATES['seniority']},
            {"role": "user", "content": description}
        ]
        return get_model_response(model_qwen, tokenizer_qwen, messages)

    else:
        return "Unrecognized question type. Please select salary, work arrangement, or seniority."

# Gradio UI
def main():
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
    ).set(
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        input_border_width="2px",
        input_border_color="*neutral_200",
    )

    with gr.Blocks(title="Job Info Analyzer", theme=theme, css="""
        .gr-radio { margin-bottom: 16px; }
        .gr-textbox { border-radius: 8px !important; padding: 12px; }
        .gr-button { border-radius: 8px; padding: 12px 24px; }
        .gr-container { max-width: 800px; margin: 0 auto; padding: 24px; }
    """) as demo:
        gr.Markdown("""
        <div style="text-align: center; margin-bottom: 24px;">
            <h1 style="color: #1e3a8a; font-weight: 600;">📂 Job Info Analyzer</h1>
            <p style="color: #475569; font-size: 0.95em;">Select question type and paste job description to extract info.</p>
        </div>
        """
        )

        with gr.Row(equal_height=True):
            qtype = gr.Radio(
                choices=["salary", "work arrangement", "seniority"],
                label="Question Type",
                value="salary",
                info="Choose the analysis aspect"
            )
            description = gr.Textbox(
                label="Job Description",
                lines=5,
                placeholder="Enter job description here..."
            )
        submit_btn = gr.Button("Analyze 🔍", variant="primary")
        output = gr.Textbox(label="Result", interactive=False, lines=4)

        submit_btn.click(fn=process_input, inputs=[qtype, description], outputs=output)

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()

