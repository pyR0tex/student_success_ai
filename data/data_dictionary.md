# Data Dictionary — Project 6

## modeling_train.csv / modeling_test.csv
- `Student_ID`: anonymous synthetic student identifier.
- `Observation_Term`: synthetic term in which signals were observed.
- `Program_ID`: synthetic academic program.
- `Entry_Type`: synthetic entry pathway (`First_Year` or `Transfer`).
- `Academic_Level`: synthetic level from 1 to 4.
- `Credits_Completed`: cumulative synthetic credits completed.
- `Prior_Term_GPA`: synthetic prior-term GPA.
- `GPA_Change`: change in academic performance relative to the previous term.
- `Course_Pass_Rate_Last_2_Terms`: proportion of recent attempted courses passed.
- `Failed_Courses_Last_2_Terms`: recent count of failed courses.
- `Withdrawals_Last_2_Terms`: recent count of withdrawals.
- `Current_Course_Load_Credits`: current enrolled credit load.
- `LMS_Login_Days_4wk`: average login-day signal from the recent four-week window.
- `Assignment_Submission_Rate_4wk`: proportion of due assignments submitted.
- `Late_Assignment_Rate_4wk`: proportion submitted late.
- `Average_Assignment_Score_4wk`: recent average assignment score.
- `Days_Since_Last_LMS_Activity`: recency of LMS activity.
- `Intervention_Needed_Next_8_Weeks`: binary synthetic prediction target.

## current_students_unlabeled.csv
Same predictive fields as the modeling files, but intentionally omits the future target.

## audit_groups.csv
- `Student_ID`
- `Audit_Group`: synthetic group label for post-hoc fairness analysis only. Do not use as a model feature.

## engagement_history.csv
Weekly synthetic engagement measures for exploratory analysis.

## student_course_history.csv
- `Student_ID`, `Term_ID`, `Course_ID`, `Credits`, `Final_Grade`, `Passed`, `Withdrawn`.

## course_catalog.csv
- `Course_ID`, `Program_ID`, `Course_Level`, `Credits`, `Course_Type`, `Offering_Cadence`, `Required_Flag`, `Requirement_Group`.

## prerequisites.csv
Directed prerequisite relationship: `Prerequisite_Course_ID` must be satisfied before `Course_ID`.

## degree_requirements.csv
Defines required courses and choose-N program elective groups.

## future_course_offerings.csv
- `Course_ID`, `Term_ID`, `Season`, `Available`, `Credits`.

## advisor_cases.csv
A small set of synthetic students selected for the required end-to-end case study.
