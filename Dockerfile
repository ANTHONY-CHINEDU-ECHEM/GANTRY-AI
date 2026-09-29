FROM python:3.12

WORKDIR /app
ENV PYTHONUNBUFFERED=1 GANTRY_LLM_PROVIDER=extractive

COPY requirements.txt requirements_llm.txt install.py ./
RUN python install.py llm

COPY . .
RUN python gantry.py build

EXPOSE 8000
CMD ["python", "gantry.py", "serve", "host=0.0.0.0", "port=8000"]
