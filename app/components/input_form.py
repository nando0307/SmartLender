"""Loan application form builder for Streamlit."""
import streamlit as st
import pandas as pd


US_STATES = [
    'AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID',
    'IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC',
    'ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD',
    'TN','TX','UT','VA','VT','WA','WI','WV','WY'
]

PURPOSES = [
    'debt_consolidation', 'credit_card', 'home_improvement',
    'major_purchase', 'small_business', 'car', 'medical', 'moving',
    'vacation', 'house', 'wedding', 'renewable_energy', 'educational', 'other'
]

EMP_LENGTHS = [
    '< 1 year', '1 year', '2 years', '3 years', '4 years',
    '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
]

SUB_GRADES = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]


def build_loan_form() -> pd.DataFrame:
    """
    Build a Streamlit form for loan application input.

    Returns:
        Single-row DataFrame with all features, or None if form not submitted.
    """
    with st.form("loan_application"):
        st.subheader("📋 Loan Application Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Loan Information**")
            loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=15000, step=500)
            term = st.selectbox("Term", ["36 months", "60 months"])
            purpose = st.selectbox("Purpose", PURPOSES)
            installment = st.number_input("Monthly Payment ($)", min_value=0.0, max_value=2000.0, value=450.0, step=10.0)
            grade = st.selectbox("Grade", list("ABCDEFG"))
            sub_grade = st.selectbox("Sub-Grade", SUB_GRADES)
            initial_list_status = st.selectbox("Listing Status", ["w", "f"])

        with col2:
            st.markdown("**Borrower Profile**")
            annual_inc = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=65000, step=1000)
            emp_length = st.selectbox("Employment Length", EMP_LENGTHS)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            verification_status = st.selectbox("Verification", ["Not Verified", "Verified", "Source Verified"])
            application_type = st.selectbox("Application Type", ["Individual", "Joint App"])
            addr_state = st.selectbox("State", US_STATES)

        with col3:
            st.markdown("**Credit Profile**")
            dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=50.0, value=18.0, step=0.5)
            open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=8)
            revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=200000, value=15000, step=500)
            revol_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=55.0, step=0.5)
            total_acc = st.number_input("Total Credit Lines", min_value=0, max_value=100, value=25)
            pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0)
            mort_acc = st.number_input("Mortgage Accounts", min_value=0, max_value=20, value=1)
            pub_rec_bankruptcies = st.number_input("Bankruptcies", min_value=0, max_value=5, value=0)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        data = {
            'loan_amnt': loan_amnt,
            'annual_inc': annual_inc,
            'dti': dti,
            'open_acc': open_acc,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'total_acc': total_acc,
            'installment': installment,
            'pub_rec': pub_rec,
            'mort_acc': mort_acc,
            'pub_rec_bankruptcies': pub_rec_bankruptcies,
            'term': term,
            'grade': grade,
            'sub_grade': sub_grade,
            'emp_length': emp_length,
            'home_ownership': home_ownership,
            'verification_status': verification_status,
            'purpose': purpose,
            'addr_state': addr_state,
            'initial_list_status': initial_list_status,
            'application_type': application_type,
        }
        return pd.DataFrame([data])

    return None
