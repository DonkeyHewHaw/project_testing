FROM python:3.10
COPY . /app
WORKDIR /app
RUN python -m venv venv
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["sh", "-c", ". venv/bin/activate && streamlit run app.py"]