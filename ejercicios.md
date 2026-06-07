1. Para el modelo, tenemos varias opciones, como "deepseek/deepseek-v4-flash", "openai/o4-mini", "google/gemini-3.1-flash-lite", entre otros. En este ejemplo, usaremos "deepseek/deepseek-v4-flash". ¿Por qué elegimos este modelo? ¿Qué tamaño de contexto tiene? ¿Qué precio tiene?  ¿Qué latencia tiene? 

2. Prueba a ingerir un documento con información en Euskera. ¿Qué modelo se comporta mejor?

3. Prueba el pdf_loader con un pdf que tenga tablas. ¿Cómo se comporta? ¿Qué modelo se comporta mejor?

4. Queremos mantener las URLs de noticias actualizadas (cuando el ayuntamiento de Donostia publique nuevas noticias, queremos que se añadan a nuestra base de datos vectorial). ¿Cómo lo harías?

5. Con el comando:

    ```
    python test_load_pdf.py  ./cas-plan-donostia-gazteria-2025-2027.pdf
    ```

   El script test_load_pdf.py carga el PDF y lo divide en fragmentos. ¿Cuántos fragmentos se crean? ¿Cuál es el tamaño de cada fragmento? ¿Cómo se determina el tamaño de los fragmentos? 

6. Basándote en el script bare_minimum.py, modifica test_load_pdf.py para que guarde los documentos en una base de datos vectorial ChromaDB (en la colección plan_donostia_gazteria) utilizando el modelo "deepseek/deepseek-v4-flash". Prueba a lanzar la consulta "¿Cuál es la misión del plan Gazteria?" para comprobar si el modelo responde correctamente.

7. El fichero PDF Plan DONOSTIA GAZTERIA 2025-2027 (cas-plan-donostia-gazteria-2025-2027.pdf) proviene de https://www.donostia.eus/documents/d/asset-library-100431/cas-plan-donostia-gazteria-2025-2027. Sin embargo, el script de ingestión pdf_loader.py no guarda la URL de origen del documento. ¿Cómo modificarías el script para que guarde esta información?

8. Queremos crear un chatbot para responder preguntas sobre el plan DONOSTIA GAZTERIA 2025-2027 utilizando el modelo "deepseek/deepseek-v4-flash". 



