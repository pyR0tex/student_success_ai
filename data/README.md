# Project 6 Student Success Early-Signal & Academic Planning System

This package contains a fully synthetic dataset for an AI course project. No real students, institutions, courses, or academic records are represented.

## Core task
Build and evaluate (1) an early-signal model and (2) a constraint-based academic planning component. The optional agentic layer is not required.

## Scale
- Synthetic students: 2200
- Programs: 4
- Courses: 64
- Historical labeled modeling rows: 6600
- Training rows: 5148
- Test rows: 1452
- Current unlabeled rows: 2200
- Weekly engagement rows: 52800
- Course-history rows: 14709
- Advisor case studies: 12

## Files
- `students.csv`: current academic profile
- `modeling_train.csv`: labeled historical observations for model development
- `modeling_test.csv`: labeled historical observations for final model evaluation
- `current_students_unlabeled.csv`: current-term observations for advisor-facing predictions
- `audit_groups.csv`: synthetic audit groups for fairness analysis ONLY; do not use as a predictive feature
- `engagement_history.csv`: weekly engagement details
- `student_course_history.csv`: historical course outcomes
- `course_catalog.csv`: synthetic course catalog
- `prerequisites.csv`: prerequisite edges
- `degree_requirements.csv`: program degree requirements
- `future_course_offerings.csv`: availability for the next two synthetic terms
- `advisor_cases.csv`: selected synthetic students for required case-study analysis
- `data_dictionary.md`: field definitions and usage notes

## Important modeling rule
`Audit_Group` is intentionally kept in a separate file. It exists only to evaluate whether model errors differ across synthetic groups. It must not be used as a feature.

## Important interpretation rule
The target means that, in the synthetic data-generating process, a student required a meaningful academic intervention in the following eight weeks. It is an educational label, not a diagnosis and not a statement about a real person.
