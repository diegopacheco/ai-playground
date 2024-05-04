import streamlit as st
import pandas as pd
 
st.write("""
# My first app
Hello *world!*
""")

# generate a pandas dataframe in memory
df = pd.DataFrame({
  'days': [1, 2, 3, 4],
  'rain_ammout': [10, 20, 30, 40]
})
st.line_chart(df)