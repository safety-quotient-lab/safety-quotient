#!/usr/bin/env python3
"""
Generate AD (authority_dynamics) augmentation batch — 500 texts.

Covers 0-10 scale with emphasis on:
  - Score 0-2: formal authority texts (coercive, unilateral, punitive)
  - Score 8-10: consensus governance texts (collaborative, egalitarian)
  - Score 3-7: mixed formal organizational texts

Source: synthetic generation (controlled prompts), per §69 sourcing policy.
Output: data/ad-augmentation-batch.jsonl (one JSON record per line)
Run: python3 scripts/generate_ad_batch.py
"""

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "ad-augmentation-batch.jsonl"


# ──────────────────────────────────────────────────────────────────────────────
# SCORE 0-1: Extreme authority abuse — punitive, no-recourse, dehumanizing
# ──────────────────────────────────────────────────────────────────────────────
texts_score_0_1 = [
    {"text": "Effective immediately, all employees who fail to meet the revised quota will be terminated without notice or severance. There is no appeal process. This decision is final and non-negotiable.", "source": "synthetic-ad-augmentation"},
    {"text": "You are hereby ordered to cease all unauthorized activities and report to disciplinary processing within 24 hours. Non-compliance will result in immediate detention and referral for criminal prosecution.", "source": "synthetic-ad-augmentation"},
    {"text": "All workers on the floor are prohibited from speaking to one another during shifts. Violations will be logged and used as grounds for immediate dismissal. Supervisors have been authorized to enforce this rule with summary penalties.", "source": "synthetic-ad-augmentation"},
    {"text": "Management reserves the absolute right to modify job duties, compensation, or working conditions at any time without prior notice or employee consent. Continued employment constitutes acceptance of any such changes.", "source": "synthetic-ad-augmentation"},
    {"text": "By order of the Director: employees found to have discussed salary information with colleagues will face immediate disciplinary action up to and including termination. This policy is not subject to grievance procedures.", "source": "synthetic-ad-augmentation"},
    {"text": "As of this memorandum, remote work arrangements are discontinued. All personnel must report to the office seven days per week or face disciplinary consequences. No exceptions will be granted.", "source": "synthetic-ad-augmentation"},
    {"text": "The following individuals have been identified for mandatory retraining. Failure to complete retraining within the prescribed timeline will result in demotion or termination. Attendance is not optional.", "source": "synthetic-ad-augmentation"},
    {"text": "Your employment is contingent upon compliance with all directives issued by senior management, regardless of whether those directives conflict with previously agreed terms. Refusal to comply is grounds for immediate termination.", "source": "synthetic-ad-augmentation"},
    {"text": "Unit Command Order 14-A: All personnel are to surrender personal electronic devices before entering the facility. Devices will be held indefinitely at command discretion. No receipts will be issued.", "source": "synthetic-ad-augmentation"},
    {"text": "This department will operate under a zero-tolerance policy for dissent. Any employee found questioning, challenging, or refusing to implement managerial decisions will be subject to immediate disciplinary review.", "source": "synthetic-ad-augmentation"},
    {"text": "Directive 7: All contractors are to be treated as at-will employees with no protections. Contracts are void upon managerial determination that performance is unsatisfactory. No cure period applies.", "source": "synthetic-ad-augmentation"},
    {"text": "Employees are not permitted to organize, form associations, or engage in collective activities of any kind. Such activities will be treated as insubordination and result in termination.", "source": "synthetic-ad-augmentation"},
    {"text": "The company has determined that your position is redundant. You are required to vacate your office by close of business today and return all company property immediately. Security will escort you from the premises.", "source": "synthetic-ad-augmentation"},
    {"text": "All performance targets have been revised upward by 40%. Employees who fail to meet the new targets for two consecutive weeks will be placed on a performance improvement plan with termination as the presumed outcome.", "source": "synthetic-ad-augmentation"},
    {"text": "By authority of the Board: no employee shall disclose any information about working conditions, compensation, disciplinary procedures, or company policies to external parties. Violations will result in civil and criminal liability.", "source": "synthetic-ad-augmentation"},
    {"text": "Mandatory overtime is in effect for all staff until further notice. Refusal to work overtime will be recorded as insubordination. No additional compensation will be provided.", "source": "synthetic-ad-augmentation"},
    {"text": "It has come to our attention that some employees have been taking breaks in excess of the allotted five minutes. Effective today, breaks are eliminated. Supervisors have authority to dock pay for non-compliance.", "source": "synthetic-ad-augmentation"},
    {"text": "This is a final written warning. Any further infractions, however minor, will result in immediate termination without further notice. You have no right to contest this determination.", "source": "synthetic-ad-augmentation"},
    {"text": "General Order: All personnel are to obey instructions from superior officers without question or delay. Requests for explanation or justification are not to be made during operational hours.", "source": "synthetic-ad-augmentation"},
    {"text": "The reorganization will proceed as announced. Employees who are not reselected for their roles will be terminated. No internal transfers will be permitted. This decision is not subject to review.", "source": "synthetic-ad-augmentation"},
    {"text": "Administrative Directive 22: All leave requests are suspended until further notice. Emergency leave may be requested through the Deputy Director, who has sole discretion over approval. No appeal is available.", "source": "synthetic-ad-augmentation"},
    {"text": "You are required to sign the enclosed arbitration waiver as a condition of continued employment. Refusal to sign will be treated as voluntary resignation effective immediately.", "source": "synthetic-ad-augmentation"},
    {"text": "Senior management has determined that employee survey responses were unsatisfactory. As a consequence, bonuses for all staff are cancelled this quarter. Individual performance is not a factor in this decision.", "source": "synthetic-ad-augmentation"},
    {"text": "All access to external websites is hereby revoked. Internet use is restricted to company-approved tools only. Circumventing this restriction, even for work purposes, will result in immediate termination.", "source": "synthetic-ad-augmentation"},
    {"text": "The Chief Operating Officer has ordered a 20% reduction in workforce. Affected employees will be notified by end of day. There is no buyout, no transition assistance, and no appeals process.", "source": "synthetic-ad-augmentation"},
    {"text": "Enforcement Notice: Residents who fail to vacate the premises by the specified date will be subject to immediate legal action, including criminal trespass charges. No extensions will be granted.", "source": "synthetic-ad-augmentation"},
    {"text": "This regulation takes effect immediately upon publication. No comment period will be provided. Regulated entities must achieve full compliance within 30 days or face civil monetary penalties.", "source": "synthetic-ad-augmentation"},
    {"text": "Order of the Commandant: Personnel are prohibited from communicating with media, family members, or legal representatives regarding unit operations. Violations are subject to court-martial.", "source": "synthetic-ad-augmentation"},
    {"text": "The new attendance tracking system will monitor all keystrokes and movements throughout the workday. This monitoring is non-negotiable and constitutes a condition of employment.", "source": "synthetic-ad-augmentation"},
    {"text": "Effective this cycle, all decisions regarding client assignments rest exclusively with the Regional Director. Staff preferences and prior arrangements are superseded. No discussion will be entertained.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# SCORE 2: Authority abuse — punitive and unilateral but marginally less extreme
# ──────────────────────────────────────────────────────────────────────────────
texts_score_2 = [
    {"text": "All personnel are required to submit daily activity logs for supervisor review. Logs must be submitted by 4:30 PM. Late submissions will result in formal disciplinary notation in employee files.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization has adopted a strict dress code effective next Monday. Non-compliance will result in employees being sent home without pay. Managers have authority to enforce this policy at their discretion.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff are reminded that the company's intellectual property policy applies to all work conducted on personal time if the work relates to the company's core business. Violations may result in legal action.", "source": "synthetic-ad-augmentation"},
    {"text": "Management has decided to eliminate the flexible scheduling program. All employees must adhere to the standard 8-5 schedule. Requests for accommodation will be considered on an exceptional basis only.", "source": "synthetic-ad-augmentation"},
    {"text": "Section 4.7: Employees who receive more than two warnings in a 90-day period are subject to immediate termination. Warning thresholds may be adjusted by management at any time. Employee consent is not required.", "source": "synthetic-ad-augmentation"},
    {"text": "As a condition of receiving severance pay, you are required to sign a comprehensive release of all claims against the company, its officers, directors, and employees. You have 48 hours to sign.", "source": "synthetic-ad-augmentation"},
    {"text": "All personnel changes, including internal transfers, must be approved by the Chief Executive Officer. Department heads no longer have authority to approve transfers independently.", "source": "synthetic-ad-augmentation"},
    {"text": "This directive overrides all prior guidance on customer service protocols. Employees must follow the new script verbatim. Deviation from the script, even if beneficial to the customer, is prohibited.", "source": "synthetic-ad-augmentation"},
    {"text": "Compliance is required within seven business days. The agency will not entertain requests for extensions. Failure to comply will result in license revocation and referral for prosecution.", "source": "synthetic-ad-augmentation"},
    {"text": "The wage freeze will remain in effect for a minimum of 18 months. This decision was made by the Board of Directors and is not subject to renegotiation.", "source": "synthetic-ad-augmentation"},
    {"text": "Employees are prohibited from using personal mobile devices on company premises at any time. Devices must be stored in the designated lockers. This policy will be enforced through random inspections.", "source": "synthetic-ad-augmentation"},
    {"text": "The performance management system has been revised. All employees will now be ranked against peers. The bottom 10% will be placed on performance improvement plans, regardless of their absolute performance.", "source": "synthetic-ad-augmentation"},
    {"text": "You have been assigned to the overnight shift effective next pay period. This is a business decision and is not negotiable. If you are unable to comply, you may submit a resignation.", "source": "synthetic-ad-augmentation"},
    {"text": "All hiring decisions are frozen pending executive review. Department managers do not have authority to extend offers or commitments of any kind. Violations will be treated as a serious breach of policy.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff are advised that the non-compete clause in your employment agreement applies for 36 months post-employment and covers a 200-mile radius. Enforcement is at the company's discretion.", "source": "synthetic-ad-augmentation"},
    {"text": "Regional headquarters has determined that all branch budgets will be reduced by 30% next fiscal year. Branch managers have no discretion in this matter. Efficiency will be assessed against new targets.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization will be implementing random drug testing effective immediately. Refusal to submit to testing is treated as a positive result and will result in termination.", "source": "synthetic-ad-augmentation"},
    {"text": "Any employee who files a complaint with an external regulatory agency without first exhausting internal channels will be subject to disciplinary action. Internal reporting is mandatory before external reporting.", "source": "synthetic-ad-augmentation"},
    {"text": "Client relationship management responsibilities are being transferred to the national accounts team effective immediately. Individual contributors are no longer permitted to contact named clients directly.", "source": "synthetic-ad-augmentation"},
    {"text": "The performance bonus pool has been redistributed by senior leadership. Individual bonuses may differ significantly from what was previously communicated. This decision is final.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# SCORE 3: Hierarchical with some nominal processes — clearly authority-biased
# ──────────────────────────────────────────────────────────────────────────────
texts_score_3 = [
    {"text": "All expenditures above $500 require Vice President approval. Requests must be submitted through the finance portal. Approvals are at the Vice President's discretion and will not be reviewed.", "source": "synthetic-ad-augmentation"},
    {"text": "Changes to client contracts must be approved by the Legal Department before implementation. Client requests for amendments will be evaluated by Legal on a case-by-case basis.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization's strategic priorities are set by the Executive Committee. Department heads are responsible for aligning their teams with these priorities. Staff feedback on strategic direction is not solicited at this stage.", "source": "synthetic-ad-augmentation"},
    {"text": "All communications with the media must be routed through the Communications Director. Staff should not speak to reporters under any circumstances, even in personal capacity on organizational matters.", "source": "synthetic-ad-augmentation"},
    {"text": "Performance evaluations are completed by direct supervisors. There is a formal review process, but final ratings are determined by the supervisor and ratified by the department head. Employee input is considered but not determinative.", "source": "synthetic-ad-augmentation"},
    {"text": "The reorganization plan was developed by senior management over a six-month period. While the plan is now final, managers are encouraged to surface operational concerns through normal reporting channels.", "source": "synthetic-ad-augmentation"},
    {"text": "Project assignments are made by the Program Director based on organizational needs. Individual preferences are taken into consideration where possible, but operational requirements take precedence.", "source": "synthetic-ad-augmentation"},
    {"text": "The new benefits package has been finalized by the Benefits Committee. The package will be implemented as described. Employees may submit questions through the HR portal.", "source": "synthetic-ad-augmentation"},
    {"text": "Emergency procedures are established by the Safety Committee. All staff are required to follow these procedures without modification. The Committee will review procedures annually.", "source": "synthetic-ad-augmentation"},
    {"text": "Salary grades and bands are set by the Compensation Team. While managers may advocate for their direct reports, final compensation decisions rest with the Compensation Team and are not subject to negotiation.", "source": "synthetic-ad-augmentation"},
    {"text": "Access to restricted research data requires written approval from the Principal Investigator. Requests will be reviewed on a need-to-know basis. Decisions are final and not appealable.", "source": "synthetic-ad-augmentation"},
    {"text": "The Board has approved a new expense reimbursement policy. The Finance Committee will administer the policy. Exceptions require documentation and CFO approval.", "source": "synthetic-ad-augmentation"},
    {"text": "Travel must be pre-approved by department heads using the standard approval form. Travel that exceeds the approved budget must be justified post-trip. Repeated overages will be flagged for audit.", "source": "synthetic-ad-augmentation"},
    {"text": "All IT system changes must be approved by the IT Security Committee. Unauthorized modifications are prohibited. Violations will be reported to the CISO.", "source": "synthetic-ad-augmentation"},
    {"text": "The curriculum for mandatory training has been set by the Learning and Development team. Completion is required and tracked. Feedback on training quality may be submitted through the evaluation form.", "source": "synthetic-ad-augmentation"},
    {"text": "Promotion decisions are made by the Talent Committee based on performance data, manager recommendations, and organizational need. Employees are not present during these discussions.", "source": "synthetic-ad-augmentation"},
    {"text": "The schedule for facility maintenance is determined by the Operations team. Departments may flag urgent needs through the ticketing system, but scheduling priority rests with Operations.", "source": "synthetic-ad-augmentation"},
    {"text": "Content published on official channels must be reviewed and approved by the Director of Communications before release. Editorial decisions are the Director's responsibility.", "source": "synthetic-ad-augmentation"},
    {"text": "Research priorities for the coming fiscal year have been established by the Scientific Advisory Board. Principal investigators are expected to align grant proposals with these priorities.", "source": "synthetic-ad-augmentation"},
    {"text": "Supply chain vendor selections are finalized by the Procurement Committee. Departmental input is solicited prior to finalization but the Committee has binding authority over vendor choices.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# SCORE 4: Authority-biased with some process — nominal input channels exist
# ──────────────────────────────────────────────────────────────────────────────
texts_score_4 = [
    {"text": "The new policy was developed with input from department heads and is now in effect. Employees with concerns should speak with their managers, who will escalate significant issues to Human Resources.", "source": "synthetic-ad-augmentation"},
    {"text": "The Executive Team has made a final decision to consolidate the regional offices. We recognize this may be disruptive. Staff with questions about the transition should contact their supervisors.", "source": "synthetic-ad-augmentation"},
    {"text": "Following an internal review, the organization has adopted a new code of conduct. All staff are required to complete training and sign the updated acknowledgment form by the end of the month.", "source": "synthetic-ad-augmentation"},
    {"text": "Budget allocations for next year have been finalized. Department heads received advance notice and had the opportunity to provide input through the budget consultation process completed last quarter.", "source": "synthetic-ad-augmentation"},
    {"text": "This standard operating procedure is effective upon publication. Departments may submit comments or identified issues through the Standard Process Review Committee within 60 days.", "source": "synthetic-ad-augmentation"},
    {"text": "The new product line has been approved by senior leadership. Sales teams should familiarize themselves with the specifications. Feedback from sales representatives will be collected at the next quarterly meeting.", "source": "synthetic-ad-augmentation"},
    {"text": "Departmental restructuring was announced today by the Chief Operations Officer. Affected staff will be notified individually. A Q&A session will be held next Thursday for all staff to ask questions.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization is moving to a new performance management platform. Training sessions will be offered to all staff. Feedback on the platform can be submitted through the HR business partner.", "source": "synthetic-ad-augmentation"},
    {"text": "As part of the strategic plan approved by the Board, certain programs will be discontinued. Leaders are asked to communicate the decision to their teams professionally and direct concerns to HR.", "source": "synthetic-ad-augmentation"},
    {"text": "The revised procurement guidelines are now in force. Vendors selected outside these guidelines will require exception approval. The Procurement Lead can advise on the exception process.", "source": "synthetic-ad-augmentation"},
    {"text": "Following an external audit, the organization will implement enhanced monitoring of financial transactions. Finance staff have been briefed. All staff should be aware that transactions are subject to review.", "source": "synthetic-ad-augmentation"},
    {"text": "The leadership team has decided to move client-facing deliverables to a new timeline. Account managers are expected to communicate the new schedule to clients and manage expectations accordingly.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization will adopt a new data classification policy. All staff are responsible for compliance. The Data Governance team will run information sessions over the next two weeks.", "source": "synthetic-ad-augmentation"},
    {"text": "An organizational effectiveness review has been completed. Based on the findings, two teams will be merged under new leadership. Affected managers will be notified by the end of the week.", "source": "synthetic-ad-augmentation"},
    {"text": "The new vendor contract has been executed. Departments that relied on the previous vendor should work with the Operations team to manage the transition according to the provided timeline.", "source": "synthetic-ad-augmentation"},
    {"text": "Effective Q3, the organization will standardize all reporting formats. The analytics team has designed the new templates. Staff should attend one of the scheduled briefing sessions before the deadline.", "source": "synthetic-ad-augmentation"},
    {"text": "Leadership has determined that the office space will be reconfigured to support a hybrid work model. Seating will be assigned based on team structures. Space allocation details will be shared next week.", "source": "synthetic-ad-augmentation"},
    {"text": "The policy on outside employment has been updated. Staff are advised to review the new policy carefully. Requests for exceptions should be submitted in writing to the HR Director.", "source": "synthetic-ad-augmentation"},
    {"text": "The Chief Information Officer has approved a new cyber security protocol. All staff must complete the associated training by month's end. Non-compliance will be escalated to department heads.", "source": "synthetic-ad-augmentation"},
    {"text": "Following review of the customer satisfaction data, leadership has determined that service delivery standards will be revised. The Customer Experience team will issue updated guidelines to all client-facing staff.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# SCORE 5: Neutral — neither coercive nor egalitarian
# ──────────────────────────────────────────────────────────────────────────────
texts_score_5 = [
    {"text": "The quarterly planning cycle begins next month. Department heads should prepare budget proposals following the standard template and submit them to Finance by the 15th.", "source": "synthetic-ad-augmentation"},
    {"text": "As part of routine operations, the system will be offline for maintenance from 10 PM to 2 AM on Saturday. Users should save their work before that window.", "source": "synthetic-ad-augmentation"},
    {"text": "The new project kickoff meeting is scheduled for Tuesday at 2 PM. All project team members are expected to attend. The agenda has been shared via email.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization will be closed on the national holiday. Staff may use banked time or take the day without pay, per the standard leave policy. Please update your calendars accordingly.", "source": "synthetic-ad-augmentation"},
    {"text": "Reminder: expense reports for Q2 are due by July 31. Please submit all receipts and complete the standard form before that date to ensure timely reimbursement.", "source": "synthetic-ad-augmentation"},
    {"text": "The annual performance review process starts next month. Supervisors should schedule one-on-one meetings with their direct reports in preparation. The HR portal will have updated guidance by the end of the week.", "source": "synthetic-ad-augmentation"},
    {"text": "Standard security badges will be reissued to all staff at the end of the year. This is a routine renewal. Staff should continue using their current badges until notified otherwise.", "source": "synthetic-ad-augmentation"},
    {"text": "The updated employee handbook is available on the intranet. All staff should review the revised sections before the end of the month. Please acknowledge receipt via the HR portal.", "source": "synthetic-ad-augmentation"},
    {"text": "The annual fire drill is scheduled for next Wednesday at 10 AM. Please follow the posted evacuation routes and gather at the designated assembly point.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff are reminded that personal leave requests for the holiday period should be submitted at least four weeks in advance. Approval is subject to operational requirements and is reviewed by direct supervisors.", "source": "synthetic-ad-augmentation"},
    {"text": "The IT helpdesk will be transitioning to a new ticketing system next quarter. Training materials will be distributed in advance. The transition is expected to be seamless for end users.", "source": "synthetic-ad-augmentation"},
    {"text": "Onboarding schedules for new hires starting in September have been finalized. Hiring managers should connect with HR to ensure all required documentation is in order before the start date.", "source": "synthetic-ad-augmentation"},
    {"text": "The client portal will be updated to version 3.2 on Thursday evening. The new version includes minor usability improvements. No data migration is required.", "source": "synthetic-ad-augmentation"},
    {"text": "Parking permits for the next calendar year are now available. Staff who wish to renew or apply should complete the online form. Permits are issued on a first-come, first-served basis.", "source": "synthetic-ad-augmentation"},
    {"text": "The annual benefits enrollment period opens on October 1. Staff should review their current elections and make changes if needed. Enrollment materials will be sent to personal email addresses on file.", "source": "synthetic-ad-augmentation"},
    {"text": "The research team is preparing the annual report. Contributors should submit their sections to the editor by September 30. The final report will be distributed to all stakeholders in November.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff are invited to participate in the annual workplace survey. Responses are anonymous. Results will be shared with all staff and used to inform planning.", "source": "synthetic-ad-augmentation"},
    {"text": "The vendor payment schedule has been updated. Accounts payable will process approved invoices on the 1st and 15th of each month. Please ensure invoices are approved at least three business days in advance.", "source": "synthetic-ad-augmentation"},
    {"text": "A new digital collaboration tool has been deployed to all teams. The product team will host optional introduction sessions next week. Recorded walkthroughs will also be available on the intranet.", "source": "synthetic-ad-augmentation"},
    {"text": "Monthly team meetings will continue on the first Monday of each month. The agenda format remains the same. Please submit agenda items to the team coordinator by the Thursday prior.", "source": "synthetic-ad-augmentation"},
    {"text": "End-of-year inventory checks are scheduled for December 15. Department leads should assign staff to the inventory team and coordinate with the warehouse manager on timing.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization's conflict of interest policy requires annual disclosure. All staff should complete the disclosure form by the end of the fiscal year. Instructions are on the compliance portal.", "source": "synthetic-ad-augmentation"},
    {"text": "Travel pre-approval requests should be submitted at least two weeks before the travel date. Reimbursement requests with required receipts should be submitted within 30 days of return.", "source": "synthetic-ad-augmentation"},
    {"text": "The facilities team will be conducting the annual workplace inspection next week. Staff should ensure their workspaces comply with safety guidelines before the inspection date.", "source": "synthetic-ad-augmentation"},
    {"text": "Project closeout documentation for Phase 2 is due to the Program Manager by end of month. Standard templates are available in the project management system.", "source": "synthetic-ad-augmentation"},
    {"text": "All training completions must be recorded in the learning management system by the employee. Supervisors should verify completion rates for their teams and follow up on any gaps.", "source": "synthetic-ad-augmentation"},
    {"text": "The finance team will be migrating to the new accounting platform in January. Current staff should ensure all pending work in the legacy system is finalized by December 20.", "source": "synthetic-ad-augmentation"},
    {"text": "Technical staff should follow the established change management process for all production deployments. Changes outside the process require Change Advisory Board review.", "source": "synthetic-ad-augmentation"},
    {"text": "All data processing agreements with external vendors are subject to legal review before signing. The procurement team should coordinate with legal counsel at least 10 business days before any signing deadline.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization's social media policy applies to all staff. Employees representing the organization on personal channels should review the guidelines available on the intranet.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# SCORE 6: Mildly positive — some participation, respectful authority
# ──────────────────────────────────────────────────────────────────────────────
texts_score_6 = [
    {"text": "The department head has asked all team members to review the draft strategic plan and submit written feedback before the all-hands meeting. Your input will shape the final version.", "source": "synthetic-ad-augmentation"},
    {"text": "Before finalizing the new workflow, the operations team will hold open office hours this week for staff to raise questions or concerns. All feedback will be reviewed and responded to.", "source": "synthetic-ad-augmentation"},
    {"text": "The leadership team values your perspective on the upcoming changes. Staff are encouraged to share their views through the anonymous feedback channel, which will remain open throughout the transition.", "source": "synthetic-ad-augmentation"},
    {"text": "The revised maternity and paternity leave policy was developed with input from the Employee Resource Group and is being shared for staff review before formal adoption.", "source": "synthetic-ad-augmentation"},
    {"text": "Managers are encouraged to involve their teams in setting team-level goals that align with organizational priorities. How teams achieve those goals is largely within their discretion.", "source": "synthetic-ad-augmentation"},
    {"text": "We are making a significant change to how we handle client escalations. Before implementation, we'd like all client-facing staff to participate in a 30-minute walkthrough and Q&A.", "source": "synthetic-ad-augmentation"},
    {"text": "The research direction for the next grant cycle was developed through consultation with both senior researchers and early-career staff. A summary of the process is attached.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff may raise concerns about any organizational decision through the ombudsperson, who operates independently of management. This channel is confidential and does not require prior escalation.", "source": "synthetic-ad-augmentation"},
    {"text": "The Board approved the new strategic plan after extensive consultation with staff, clients, and community partners over the past eight months. A summary of how feedback shaped the plan is available on the website.", "source": "synthetic-ad-augmentation"},
    {"text": "Team leads are given considerable discretion in how they structure their team's work. Organizational requirements include delivery timelines and quality standards; how teams get there is their call.", "source": "synthetic-ad-augmentation"},
    {"text": "The HR department has established a formal appeal process for disciplinary decisions. All staff have the right to appeal with a designated independent reviewer within 30 days of a formal action.", "source": "synthetic-ad-augmentation"},
    {"text": "Before the rollout, we conducted a pilot with volunteers from each department to identify usability issues. The product has been significantly improved based on their input.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff working on cross-functional projects have significant latitude in how they coordinate their work. Reporting requirements are minimal and focused on outcomes rather than methods.", "source": "synthetic-ad-augmentation"},
    {"text": "The annual review process includes a self-assessment that carries meaningful weight in the final evaluation. Supervisors are asked to engage genuinely with self-assessments before finalizing ratings.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization holds a formal Town Hall every quarter where any staff member can submit questions. All questions are answered on the record. Recordings are shared with staff who cannot attend.", "source": "synthetic-ad-augmentation"},
    {"text": "Department heads have been delegated authority to make hiring decisions for positions below Director level. This delegation is intended to accelerate decision-making and respect team expertise.", "source": "synthetic-ad-augmentation"},
    {"text": "The new workspace design was informed by a staff preference survey, focus groups with representatives from each department, and review of industry research on hybrid work environments.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff who disagree with a performance rating have a formal right to request a second-level review. The review process is documented and the decision of the second reviewer is binding.", "source": "synthetic-ad-augmentation"},
    {"text": "The leadership team makes significant decisions through a defined process that includes input from middle management and senior individual contributors before approval.", "source": "synthetic-ad-augmentation"},
    {"text": "All project proposals are reviewed by a cross-functional committee that includes representation from frontline staff. The committee's recommendation is forwarded to the sponsoring executive.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# SCORE 7: Collaborative — meaningful participation, authority used transparently
# ──────────────────────────────────────────────────────────────────────────────
texts_score_7 = [
    {"text": "The strategic planning committee includes staff representatives from every level of the organization. Their recommendations are taken seriously and have shaped major decisions over the past three years.", "source": "synthetic-ad-augmentation"},
    {"text": "Before any significant policy change is finalized, affected teams are consulted in a structured process that requires documented responses to all substantive concerns raised.", "source": "synthetic-ad-augmentation"},
    {"text": "Leadership decisions on organizational priorities are made transparent through regular all-staff briefings that include the full reasoning behind each decision, not just the conclusion.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff have a formal and effective channel to challenge managerial decisions they believe are unjust. All such challenges are reviewed by an independent panel, and staff are informed of the outcome.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization operates with delegated decision-making authority. Most operational decisions are made by the people closest to the work, with escalation reserved for matters involving significant resources or risk.", "source": "synthetic-ad-augmentation"},
    {"text": "The staff council and management team co-design the organization's people policies. Both parties have the right to request reconsideration if they believe a proposed policy is unfair or unworkable.", "source": "synthetic-ad-augmentation"},
    {"text": "Performance management at this organization is a two-way process. Staff evaluate their supervisors annually, and this feedback is reviewed by the leadership team and acted upon.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization has eliminated stack ranking. Evaluations focus on individual growth and team contribution, with no mandatory distribution requirements. Staff can see the full evaluation criteria.", "source": "synthetic-ad-augmentation"},
    {"text": "When resource constraints require difficult tradeoffs, leadership holds open working sessions with affected staff to explore options together before making a final decision.", "source": "synthetic-ad-augmentation"},
    {"text": "Whistleblower protections at this organization go beyond legal requirements. Staff who raise concerns in good faith are actively protected from informal retaliation as well as formal sanctions.", "source": "synthetic-ad-augmentation"},
    {"text": "Authority to approve project proposals has been fully distributed to team level. Teams may self-authorize projects that fit within established parameters without seeking management approval.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization's salary structure is fully transparent. All employees can see the bands for every role, the criteria for advancement, and the current distribution of salaries within each band.", "source": "synthetic-ad-augmentation"},
    {"text": "All significant organizational decisions are preceded by a formal consultation period during which staff can submit comments, and management must respond to all substantive points before proceeding.", "source": "synthetic-ad-augmentation"},
    {"text": "The mentorship and sponsorship program is open to all staff, regardless of level. Eligibility is based on expressed interest and fit, not on manager recommendation.", "source": "synthetic-ad-augmentation"},
    {"text": "Team leaders are expected to surface and attribute the contributions of all team members when reporting to senior leadership. Invisibility of individual contributions is treated as a leadership failure.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization's governance framework requires that any policy affecting working conditions be co-developed with the joint labor-management committee before taking effect.", "source": "synthetic-ad-augmentation"},
    {"text": "Disputes about work assignments or performance assessments may be brought to a neutral ombudsperson. The ombudsperson's recommendations are binding, and staff are not penalized for using the channel.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization shares financial results, strategic risks, and significant decisions with all staff on the same timeline that it shares them with the Board. There is no information hierarchy.", "source": "synthetic-ad-augmentation"},
    {"text": "Leaders at this organization are evaluated partly on whether they demonstrate genuine openness to feedback, make decisions transparently, and support the professional agency of their direct reports.", "source": "synthetic-ad-augmentation"},
    {"text": "All staff, including junior contributors, are encouraged to propose process improvements through a formal channel. Accepted proposals are implemented and credited to the proposer.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# SCORE 8: Strong egalitarianism — distributed authority, mutual accountability
# ──────────────────────────────────────────────────────────────────────────────
texts_score_8 = [
    {"text": "Our collective decision-making process requires consensus from all affected parties before any significant change can be implemented. Any member may block a proposal, and the block must be resolved through dialogue.", "source": "synthetic-ad-augmentation"},
    {"text": "The cooperative is governed by its members. Every member has one vote in all governance matters, regardless of their tenure, financial contribution, or operational role.", "source": "synthetic-ad-augmentation"},
    {"text": "Leadership roles in this community rotate on an annual basis. No individual may hold the same leadership position for more than two consecutive years. Rotation is mandatory, not optional.", "source": "synthetic-ad-augmentation"},
    {"text": "All decisions that affect the whole community are made at the general assembly, where every participant has equal speaking time and equal voting weight.", "source": "synthetic-ad-augmentation"},
    {"text": "The faculty senate operates on a principle of shared governance: academic decisions are made by faculty, administrative decisions are made by administration, and matters affecting both require joint approval.", "source": "synthetic-ad-augmentation"},
    {"text": "Our contribution-based governance model ensures that those most affected by a decision have the most say in making it. Voting weights are proportional to involvement, not to formal title or seniority.", "source": "synthetic-ad-augmentation"},
    {"text": "This open-source project follows a consensus model for all technical decisions. Any contributor may raise a concern, and no decision proceeds until the concern is substantively addressed.", "source": "synthetic-ad-augmentation"},
    {"text": "Community guidelines are drafted collaboratively through an open wiki process. Any member may propose changes. Changes require a 14-day comment period and a two-thirds approval vote to take effect.", "source": "synthetic-ad-augmentation"},
    {"text": "The research team operates on a principle of credit equality: all contributors who participated substantially are listed as co-authors, and no one is listed on work they did not contribute to.", "source": "synthetic-ad-augmentation"},
    {"text": "Conflict resolution within the community is handled through a peer mediation process. Mediators are trained volunteers elected by the community, not appointed by management.", "source": "synthetic-ad-augmentation"},
    {"text": "The network's bylaws prohibit any single member from holding more than 25% of the votes on the governing board, regardless of membership or contribution size.", "source": "synthetic-ad-augmentation"},
    {"text": "All working group decisions are made through a documented consent process. Every member of the working group has the right to raise an objection, and objections are recorded and addressed.", "source": "synthetic-ad-augmentation"},
    {"text": "Compensation in this cooperative is set through a transparent, member-controlled process. The ratio between the highest and lowest compensation is capped by charter at 3:1.", "source": "synthetic-ad-augmentation"},
    {"text": "Any member of the community may initiate a governance review by submitting a proposal with at least 5% member co-signers. Governance reviews are open to all members and proceed without management veto.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization's code of conduct is enforced by a peer-elected conduct committee, not by management. The committee operates independently and its decisions are final.", "source": "synthetic-ad-augmentation"},
    {"text": "Resource allocation decisions are made through a participatory budgeting process in which every member of the community has an equal vote on how shared resources are deployed.", "source": "synthetic-ad-augmentation"},
    {"text": "Leadership effectiveness is evaluated annually through a 360-degree process in which every direct report's feedback carries equal weight to that of senior supervisors.", "source": "synthetic-ad-augmentation"},
    {"text": "Our constitution guarantees every member the right to full information about decisions that affect them, with sufficient time to respond before the decision is finalized.", "source": "synthetic-ad-augmentation"},
    {"text": "The deliberative process used by this assembly requires that every proposal be tested through structured dialogue before a vote is called, ensuring that majority preferences do not silence minority concerns.", "source": "synthetic-ad-augmentation"},
    {"text": "All community policies sunset automatically after three years. Continuation requires a fresh consensus vote. This structure ensures that no rule persists beyond the community's active endorsement.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# SCORE 9-10: Maximum equity — pure consensus, mutual recognition, power fully shared
# ──────────────────────────────────────────────────────────────────────────────
texts_score_9_10 = [
    {"text": "This community is structured on the principle of full horizontal governance. There are no permanent leaders, no hierarchical titles, and no decisions that any individual can make unilaterally. All power is held collectively.", "source": "synthetic-ad-augmentation"},
    {"text": "Our governance process requires that no decision affecting any member proceed without that member's explicit consent. The right of any person to block a decision affecting them is absolute.", "source": "synthetic-ad-augmentation"},
    {"text": "We operate through distributed authority: every working group holds full autonomy within its domain, and coordination between groups is handled through peer agreement, not central directive.", "source": "synthetic-ad-augmentation"},
    {"text": "This assembly operates by consensus, which means we keep working until everyone can genuinely live with the decision — not just the majority. A decision that leaves members in genuine distress is not a decision we have finished making.", "source": "synthetic-ad-augmentation"},
    {"text": "Our founding charter establishes that no officer, board member, or committee has the authority to bind the collective without a full membership vote. All actions taken in the collective's name require member ratification.", "source": "synthetic-ad-augmentation"},
    {"text": "Authority in this network is inherently temporary and contingent. Any mandate given to any person or body can be revoked by the community at any time, for any reason, through the standard consent process.", "source": "synthetic-ad-augmentation"},
    {"text": "We recognize that power imbalances can emerge even in intentionally egalitarian spaces. Our governance process includes a regular power audit, conducted by an elected equity team, to surface and correct structural asymmetries.", "source": "synthetic-ad-augmentation"},
    {"text": "Membership in this collective confers equal standing in all governance matters. Newcomers and founding members have identical rights. Tenure confers no additional authority.", "source": "synthetic-ad-augmentation"},
    {"text": "The network's decision-making charter commits us to ensuring that those with less institutional voice are actively enabled to participate fully, not merely formally included.", "source": "synthetic-ad-augmentation"},
    {"text": "We practice sociocracy: authority flows from role consent, not title, and any role can be adapted or dissolved by the circle that holds it, with no reference to external hierarchical authority.", "source": "synthetic-ad-augmentation"},
    {"text": "Our community agreement recognizes that meaningful participation requires material support. We provide translation, accessibility accommodations, childcare, and stipends to ensure that participation is genuinely equal, not just formally available.", "source": "synthetic-ad-augmentation"},
    {"text": "The principles of this federation prohibit any affiliate from acquiring disproportionate influence over shared governance structures. Anti-concentration provisions are embedded in the bylaws and reviewed annually.", "source": "synthetic-ad-augmentation"},
    {"text": "All deliberation in this assembly is facilitated by trained community members who are elected to role, not appointed, and who serve without authority to direct the substance of discussions.", "source": "synthetic-ad-augmentation"},
    {"text": "We commit to epistemic equity: all members' contributions to collective knowledge are treated as equally valid starting points for deliberation, regardless of formal credentials or professional status.", "source": "synthetic-ad-augmentation"},
    {"text": "Our collective agreement includes an explicit power-sharing clause: no member, regardless of their contribution or tenure, may accumulate influence that allows them to shape decisions without the community's knowledge and consent.", "source": "synthetic-ad-augmentation"},
    {"text": "This organization was founded on the principle that authority must be earned through ongoing trust, not inherited through hierarchy. We continuously re-earn the mandate to act on each other's behalf.", "source": "synthetic-ad-augmentation"},
    {"text": "All members have the right to full participation in decision-making about their own labor. No policy governing how members work may be adopted without a consent-based process that includes those whose work it governs.", "source": "synthetic-ad-augmentation"},
    {"text": "Our governance model treats the absence of objection as insufficient for consent. We require positive, affirmative agreement from those most affected before proceeding with any significant action.", "source": "synthetic-ad-augmentation"},
    {"text": "The community regularly practices structured dialogue where designated listeners document minority views and these views are integrated into the final decision record, regardless of whether they prevailed.", "source": "synthetic-ad-augmentation"},
    {"text": "Shared resources in this network are allocated through a structured community process in which every member has equal standing. No member's needs are weighted as more urgent by virtue of their role or relationship to leadership.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# Additional mid-range texts to reach target distribution
# ──────────────────────────────────────────────────────────────────────────────
texts_mid_range_additional = [
    # Score 3-4 range — additional variety
    {"text": "The procurement policy requires competitive bidding for all contracts above the threshold. Sole-source exceptions require written justification approved by the Chief Financial Officer.", "source": "synthetic-ad-augmentation"},
    {"text": "Requests for schedule exceptions must be submitted in writing to the departmental director at least 72 hours in advance. Approval is at the director's discretion.", "source": "synthetic-ad-augmentation"},
    {"text": "The staff handbook is reviewed and updated annually by HR. Staff may submit suggested updates via the HR portal during the designated review window.", "source": "synthetic-ad-augmentation"},
    {"text": "The company's bonus structure is set by the Compensation Committee and approved by the Board. The criteria for eligibility are published annually before the performance period begins.", "source": "synthetic-ad-augmentation"},
    {"text": "All communications to elected officials or regulators must be cleared through Government Affairs. Staff should route any inquiry from these parties to Government Affairs immediately.", "source": "synthetic-ad-augmentation"},
    {"text": "Research involving human subjects must receive institutional review board approval before any participant contact. IRB decisions are final and may not be appealed through departmental channels.", "source": "synthetic-ad-augmentation"},
    {"text": "Any modification to core system infrastructure must be approved by the Architecture Review Board before implementation, regardless of the urgency of the underlying need.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff requesting accommodation under applicable disability law should submit a medical certification to Human Resources. Accommodation decisions are made by HR in consultation with legal counsel.", "source": "synthetic-ad-augmentation"},
    {"text": "The academic promotion process is governed by published criteria. Tenure and promotion cases are reviewed by the departmental committee, the dean, and the provost in sequence.", "source": "synthetic-ad-augmentation"},
    {"text": "Changes to client-facing terms of service must be approved by the Legal team, the Chief Customer Officer, and the Chief Executive Officer before they are published.", "source": "synthetic-ad-augmentation"},
    # Score 5-6 range — additional variety
    {"text": "The organization has recently updated its onboarding process based on feedback from new hires. The changes aim to reduce time-to-productivity and improve early experience.", "source": "synthetic-ad-augmentation"},
    {"text": "Cross-functional teams are encouraged to self-organize where possible and to escalate coordination challenges to the PMO only when teams cannot resolve them independently.", "source": "synthetic-ad-augmentation"},
    {"text": "Performance expectations for all roles are documented in job profiles. Staff should discuss their profile with their manager at the start of each performance cycle.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization uses an internal nomination process for stretch assignments. Supervisors may nominate team members, and eligible individuals may also self-nominate with supervisory support.", "source": "synthetic-ad-augmentation"},
    {"text": "Monthly one-on-one meetings between staff and supervisors are considered standard practice. These meetings are intended to be two-way: staff should feel comfortable raising any concern.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization invites all staff to participate in the annual all-hands, where leadership presents results, strategic priorities, and addresses questions submitted in advance.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff are encouraged to identify and pursue professional development activities that align with their career goals. Requests for external training support should be submitted to HR.", "source": "synthetic-ad-augmentation"},
    {"text": "The internal job posting process allows all qualified staff to apply for open positions. Hiring managers are encouraged to give internal candidates full and fair consideration.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization runs a biannual engagement survey. Results are shared at the team level, and managers are expected to follow up with their teams on survey findings.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff who take on informal leadership roles, such as facilitating working groups or onboarding colleagues, are encouraged to document these contributions in their performance self-assessment.", "source": "synthetic-ad-augmentation"},
    # Score 6-7 range — additional variety
    {"text": "The organization commits to explaining the reasoning behind all significant management decisions in a full-staff communication within one week of the decision being made.", "source": "synthetic-ad-augmentation"},
    {"text": "Our team operates with full transparency about workload. All active projects and their owners are visible to everyone on the team. No work is assigned without the assignee's knowledge and input.", "source": "synthetic-ad-augmentation"},
    {"text": "Before finalizing the team's quarterly priorities, the manager holds a structured priority-setting session where each team member can propose, question, and advocate for items on the list.", "source": "synthetic-ad-augmentation"},
    {"text": "Interpersonal conflicts within teams are addressed through a formal peer mediation process before any manager intervention. Mediation is voluntary but strongly encouraged as a first step.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff have meaningful input into how organizational policies that affect them are designed. The policy design process includes staff review stages with documented response requirements.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization publishes the pay ranges for all roles. When a pay gap review reveals an unexplained disparity, the organization is committed to correcting it within the following compensation cycle.", "source": "synthetic-ad-augmentation"},
    {"text": "Leadership decisions about organizational structure are made through a process that includes staff representatives at the planning stage, not just announcement and implementation.", "source": "synthetic-ad-augmentation"},
    {"text": "Staff are formally empowered to decline work they believe to be unethical, without fear of retaliation. Ethics concerns should be raised first with the direct supervisor, then with the Ethics Office.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization holds open-door office hours with senior leaders every month, where any staff member can raise a question or concern directly — no pre-screening, no agenda required.", "source": "synthetic-ad-augmentation"},
    {"text": "Our organization's governance charter requires that the Board include staff-elected representatives, who hold full voting rights equivalent to those of the appointed members.", "source": "synthetic-ad-augmentation"},
    # More score 8-9 — consensus and distributed authority
    {"text": "The platform's terms of service and community guidelines are set by an elected member council, not by the platform's founding team. The founding team retains no special governance rights.", "source": "synthetic-ad-augmentation"},
    {"text": "Our association's constitution prohibits the concentration of decision-making power in any single committee or individual. Any decision that falls outside distributed mandates requires a general membership vote.", "source": "synthetic-ad-augmentation"},
    {"text": "We govern ourselves through consent, not consensus of the majority. Any member's reasoned objection can block a proposal until the concern is adequately addressed.", "source": "synthetic-ad-augmentation"},
    {"text": "Leadership in this network is facilitative, not directive. The role of network coordinators is to support members in implementing their own decisions, not to make decisions on their behalf.", "source": "synthetic-ad-augmentation"},
    {"text": "We practice double-linking: every circle sends a representative to the next level, and the next level sends a representative back. Authority flows both directions, preventing unilateral top-down control.", "source": "synthetic-ad-augmentation"},
    {"text": "The community's conflict resolution process gives the affected parties — not organizational leadership — the primary voice in designing and approving the resolution approach.", "source": "synthetic-ad-augmentation"},
    {"text": "Our collective agreement prohibits any member from speaking on behalf of the collective without explicit member authorization. No individual represents the group without a mandate.", "source": "synthetic-ad-augmentation"},
    {"text": "We use a structured equity review before adopting any new governance rule, to assess whether the rule would systematically advantage or disadvantage any group of members.", "source": "synthetic-ad-augmentation"},
    {"text": "All working group mandates are defined, extended, and ended by the membership — not by the working group itself. Groups cannot extend their own authority.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization's funding decisions are made by a member-controlled committee with rotating membership. No single funder has disproportionate influence over the committee's decisions.", "source": "synthetic-ad-augmentation"},
    # More score 0-2 — additional formal authority extremes
    {"text": "Employees found using personal email on company systems will be subject to immediate account suspension and disciplinary review. Continued use after a warning will result in termination.", "source": "synthetic-ad-augmentation"},
    {"text": "All employees must submit to biometric screening upon entering the facility. Refusal to participate will be treated as an unauthorized entry attempt and result in denial of access.", "source": "synthetic-ad-augmentation"},
    {"text": "The company reserves the right to monitor all electronic communications on company devices at any time without prior notice. By using a company device, you consent to monitoring.", "source": "synthetic-ad-augmentation"},
    {"text": "Employees on a performance improvement plan are prohibited from taking any leave, including accrued vacation, during the improvement period without express written consent of the HR Director.", "source": "synthetic-ad-augmentation"},
    {"text": "This directive is effective immediately and supersedes all prior agreements, verbal or written, regarding the subject matter. No prior arrangement will be honored unless specifically restated herein.", "source": "synthetic-ad-augmentation"},
    {"text": "The policy change was mandated by the Board. There will be no consultation period. Staff concerns should be directed to the Employee Assistance Program.", "source": "synthetic-ad-augmentation"},
    {"text": "All staff in the affected unit are required to reapply for their positions. Those not reselected will receive two weeks' notice. There is no internal transfer preference.", "source": "synthetic-ad-augmentation"},
    {"text": "The regulation was issued under emergency authority and is effective upon publication. The comment period is waived. Regulated entities must comply within 10 business days.", "source": "synthetic-ad-augmentation"},
    {"text": "Your employment is terminable at will. The company is not required to provide a reason for termination, and this at-will status cannot be modified by any verbal commitment from any company representative.", "source": "synthetic-ad-augmentation"},
    {"text": "This directive requires immediate compliance. Requests for clarification should not delay implementation. Any delay will be viewed as non-compliance and treated accordingly.", "source": "synthetic-ad-augmentation"},
    # More score 3-5 — institutional language
    {"text": "Audit findings are reported to the Audit Committee of the Board. Management is expected to respond to each finding within 30 days with a remediation plan.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization's formal dispute resolution process begins with an informal meeting between the parties and the relevant manager, followed by a formal HR review if the matter is unresolved.", "source": "synthetic-ad-augmentation"},
    {"text": "Our compliance framework requires periodic certifications from all staff in roles with significant financial or data access. Non-certifying staff will have access suspended pending resolution.", "source": "synthetic-ad-augmentation"},
    {"text": "The standard onboarding checklist includes orientation to the organization's authority structure. New hires should understand who has decision-making authority in their function from the first week.", "source": "synthetic-ad-augmentation"},
    {"text": "The organization's records retention policy sets mandatory timelines by document type. Staff responsible for records must follow the policy; exceptions require General Counsel approval.", "source": "synthetic-ad-augmentation"},
    {"text": "Procurement authority is tiered by contract value. All contracts above the senior manager threshold require director approval. All contracts above the director threshold require vice president approval.", "source": "synthetic-ad-augmentation"},
    {"text": "Project charters are approved by the sponsoring executive before project kickoff. Scope changes require a formal change request reviewed by the sponsor.", "source": "synthetic-ad-augmentation"},
    {"text": "Policy exceptions must be documented and submitted to the Policy Office. The Policy Office has 10 business days to review and will notify the requestor of its determination.", "source": "synthetic-ad-augmentation"},
    {"text": "Department heads are accountable for ensuring their teams comply with all applicable policies. Non-compliance at the team level is reflected in the department head's performance evaluation.", "source": "synthetic-ad-augmentation"},
    {"text": "All grant applications must be reviewed by the Grants Office before submission. The Grants Office will ensure alignment with organizational priorities and compliance with sponsor requirements.", "source": "synthetic-ad-augmentation"},
]

# ──────────────────────────────────────────────────────────────────────────────
# Assemble full batch
# ──────────────────────────────────────────────────────────────────────────────

all_texts = (
    texts_score_0_1 +          # 30 texts, score 0-1
    texts_score_2 +             # 20 texts, score 2
    texts_score_3 +             # 20 texts, score 3
    texts_score_4 +             # 20 texts, score 4
    texts_score_5 +             # 30 texts, score 5
    texts_score_6 +             # 20 texts, score 6
    texts_score_7 +             # 20 texts, score 7
    texts_score_8 +             # 20 texts, score 8
    texts_score_9_10 +          # 20 texts, score 9-10
    texts_mid_range_additional  # 70 texts, mixed mid-range + additional extremes
)

# Target: 270 texts (pilot batch — will expand if training confirms calibration gain)
print(f"Total texts: {len(all_texts)}")

# Validate: every entry has "text" and "source"
for i, rec in enumerate(all_texts):
    assert "text" in rec, f"Record {i} missing 'text'"
    assert "source" in rec, f"Record {i} missing 'source'"

# Write JSONL
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    for rec in all_texts:
        f.write(json.dumps(rec) + "\n")

print(f"Written: {OUTPUT_PATH}")
print(f"Score distribution (approximate targets):")
print(f"  0-1:  {len(texts_score_0_1)} texts")
print(f"  2:    {len(texts_score_2)} texts")
print(f"  3:    {len(texts_score_3)} texts")
print(f"  4:    {len(texts_score_4)} texts")
print(f"  5:    {len(texts_score_5) + 10} texts (includes 10 from mid_additional)")
print(f"  6:    {len(texts_score_6) + 10} texts (includes 10 from mid_additional)")
print(f"  7:    {len(texts_score_7) + 10} texts (includes 10 from mid_additional)")
print(f"  8:    {len(texts_score_8) + 10} texts (includes 10 from mid_additional)")
print(f"  9-10: {len(texts_score_9_10)} texts")
