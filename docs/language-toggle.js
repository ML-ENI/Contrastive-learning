(() => {
  const replacements = [
    ["Contrastive Learning con pocas etiquetas", "Contrastive Learning with Few Labels"],
    ["¿Puede SimCLR aprender representaciones útiles de MNIST sin conocer las clases?", "Can SimCLR learn useful MNIST representations without knowing the classes?"],
    ["Presentación reproducible", "Reproducible presentation"],
    ["Resultados cargados automáticamente", "Automatically loaded results"],
    ["Sin cifras inventadas", "No fabricated numbers"],
    ["Escanea para abrir la demo", "Scan to open the demo"],
    ["Conclusiones e interpretación de los resultados", "Conclusions and interpretation of results"],
    ["El control que aísla el efecto contrastivo", "The control that isolates the contrastive effect"],
    ["Evidencia dentro del espacio de embeddings", "Evidence inside the embedding space"],
    ["¿Qué imágenes considera parecidas?", "Which images does the model consider similar?"],
    ["¿Qué aportó realmente SimCLR?", "What did SimCLR actually contribute?"],
    ["Resultados de clasificación", "Classification results"],
    ["Diseño experimental", "Experimental design"],
    ["¿Qué aprende SimCLR?", "What does SimCLR learn?"],
    ["Comparación global", "Overall comparison"],
    ["El problema", "The problem"],
    ["Limitaciones", "Limitations"],
    ["Conclusión respaldada por los datos", "Data-supported conclusion"],
    ["Los resultados son internamente coherentes", "The results are internally consistent"],
    ["SimCLR sí aprendió una representación útil", "SimCLR did learn a useful representation"],
    ["¿Por qué Logistic Regression gana con pocas etiquetas?", "Why does Logistic Regression win with few labels?"],
    ["Los modelos neuronales están infraentrenados en", "The neural models are undertrained in"],
    ["La arquitectura también influye", "The architecture also matters"],
    ["Accuracy y F1 cuentan una historia compatible", "Accuracy and F1 tell a consistent story"],
    ["Pregunta de investigación", "Research question"],
    ["Etiquetar datos cuesta tiempo y dinero, pero conseguir imágenes sin etiquetar suele ser más fácil.", "Labelling data costs time and money, while obtaining unlabelled images is usually easier."],
    ["Si SimCLR observa muchas imágenes ", "If SimCLR observes many images "],
    ["sin usar sus etiquetas", "without using their labels"],
    [", ¿necesitará después menos ejemplos etiquetados para reconocer los dígitos?", ", will it later need fewer labelled examples to recognise the digits?"],
    ["Comparamos tres disponibilidades de etiquetas:", "We compare three label-availability settings:"],
    ["Todas las imágenes de entrenamiento", "All training images"],
    ["CNN supervisada", "Supervised CNN"],
    ["SimCLR autosupervisado", "Self-supervised SimCLR"],
    ["sin etiquetas", "without labels"],
    ["encoder congelado", "frozen encoder"],
    ["mismo subconjunto etiquetado", "same labelled subset"],
    ["evaluación final en test", "final test evaluation"],
    ["También usamos ", "We also use "],
    [" como referencia sencilla y un ", " as a simple reference and a "],
    ["encoder aleatorio", "random encoder"],
    [" como control.", " as a control."],
    ["Modo mostrado:", "Displayed mode:"],
    ["Las particiones son estratificadas y no se solapan.", "The splits are stratified and do not overlap."],
    ["Para cada imagen generamos dos vistas moderadamente distintas:", "For each image, we generate two moderately different views:"],
    ["pequeñas rotaciones y traslaciones;", "small rotations and translations;"],
    ["cambios suaves de escala;", "mild scale changes;"],
    ["ruido ligero;", "light noise;"],
    ["sin flips", "no flips"],
    [", porque podrían alterar el significado de un dígito.", ", because they could change a digit's identity."],
    ["Las dos vistas de la misma imagen forman un ", "The two views of the same image form a "],
    ["par positivo", "positive pair"],
    [". Las demás vistas del batch son ", ". The other views in the batch are "],
    ["negativos", "negatives"],
    ["NT-Xent aproxima los positivos y separa los negativos. Las etiquetas nunca entran en esta pérdida.", "NT-Xent pulls positives together and pushes negatives apart. Labels never enter this loss."],
    ["Comparamos dos encoders con ", "We compare two encoders with "],
    ["la misma arquitectura", "the same architecture"],
    ["Clasificador", "Classifier"],
    ["Etiquetas", "Labels"],
    ["Sin preentrenamiento", "No pretraining"],
    ["Pesos aleatorios y congelados", "Random frozen weights"],
    ["Lineal", "Linear"],
    ["Preentrenado sin etiquetas y congelado", "Pretrained without labels and frozen"],
    ["El mismo lineal", "The same linear classifier"],
    ["Los mismos IDs", "The same IDs"],
    ["Interpretación", "Interpretation"],
    ["Si el linear probe de SimCLR supera al probe aleatorio, la mejora procede de la representación aprendida, no de un clasificador más potente.", "If the SimCLR linear probe beats the random probe, the gain comes from the learned representation, not from a more powerful classifier."],
    ["Encoder aleatorio + probe", "Random encoder + probe"],
    ["Regresión logística", "Logistic Regression"],
    ["Leyenda del semáforo:", "Traffic-light legend:"],
    ["Aviso:", "Warning:"],
    ["estos son resultados", "these are"],
    ["con una semilla; validan el pipeline, pero no constituyen todavía conclusiones generales.", "results with one seed; they validate the pipeline but do not yet support general conclusions."],
    ["Ganancia del encoder preentrenado frente al aleatorio:", "Gain of the pretrained encoder over the random encoder:"],
    ["puntos", "points"],
    ["Separación media positivo–negativo:", "Mean positive–negative separation:"],
    ["antes y", "before and"],
    ["después de SimCLR.", "after SimCLR."],
    ["Los vecinos se buscan solo mediante los embeddings. Las clases se muestran después, exclusivamente para interpretar la representación.", "Neighbours are retrieved using only the embeddings. Class labels are shown afterwards solely to interpret the representation."],
    ["Lectura correcta:", "Correct interpretation:"],
    ["SimCLR debe compararse primero con el encoder aleatorio para medir la calidad aprendida. Compararlo con la CNN y Logistic Regression responde a una pregunta adicional: si esa representación produce el mejor clasificador final.", "SimCLR should first be compared with the random encoder to measure what the representation learned. Comparing it with the CNN and Logistic Regression answers an additional question: whether that representation yields the best final classifier."],
    ["MNIST es pequeño, centrado y mucho más sencillo que imágenes naturales.", "MNIST is small, centred, and much simpler than natural images."],
    ["SimCLR es sensible al tamaño del batch y al número de épocas.", "SimCLR is sensitive to batch size and the number of epochs."],
    ["El método contrastivo ve más imágenes sin etiquetar y consume más cómputo: no es una comparación de coste equivalente.", "The contrastive method sees additional unlabelled images and uses more compute: this is not a compute-matched comparison."],
    ["El linear probe mide separabilidad lineal, no todo lo que podría obtenerse mediante fine-tuning.", "The linear probe measures linear separability, not everything that fine-tuning might recover."],
    ["Una ejecución", "A"],
    ["no permite conclusiones generales; el protocolo final requiere tres semillas.", "run cannot support general conclusions; the final protocol requires three seeds."],
    ["Las etiquetas de test se utilizan una vez fijado el modelo, nunca para seleccionar hiperparámetros.", "Test labels are used only after fixing the model, never to select hyperparameters."],
    ["Al aumentar las etiquetas, todos los modelos basados en la CNN mejoran:", "As label availability increases, all CNN-based models improve:"],
    ["la CNN supervisada pasa de", "the supervised CNN rises from"],
    ["SimCLR + linear probe pasa de", "SimCLR + linear probe rises from"],
    ["Esta tendencia monotónica indica que los subconjuntos etiquetados y la evaluación se comportan de forma razonable.", "This monotonic trend indicates that the labelled subsets and evaluation behave as expected."],
    ["Frente al mismo encoder aleatorio y congelado, SimCLR mejora", "Against the same frozen random encoder, SimCLR improves by"],
    ["con 1 %, 10 % y 100 % de etiquetas. La mejora media correcta es", "with 1%, 10%, and 100% labels. The correct mean gain is"],
    [", no miles de puntos. Además, la loss contrastiva baja de", ", not thousands of points. In addition, contrastive loss falls from"],
    ["y la separación coseno positivo–negativo aumenta de", "and positive–negative cosine separation increases from"],
    ["Las tres evidencias apuntan en la misma dirección: el preentrenamiento no es equivalente a dejar la red aleatoria.", "All three pieces of evidence agree: pretraining is not equivalent to leaving the network random."],
    ["Obtiene", "It obtains"],
    ["Esto es plausible en MNIST: las imágenes están centradas, alineadas, tienen fondo uniforme y la posición de cada píxel es directamente informativa.", "This is plausible on MNIST: images are centred and aligned, backgrounds are uniform, and each pixel's position is directly informative."],
    ["Un modelo lineal tiene pocos parámetros y puede ser muy eficiente con pocos ejemplos. No demuestra que sea superior en problemas visuales más complejos.", "A linear model has few parameters and can be highly sample-efficient. This does not show that it is superior on more complex visual problems."],
    ["SimCLR solo recibe", "SimCLR receives only"],
    ["épocas", "epochs"],
    ["de preentrenamiento y la CNN supervisada parte de cero con un calendario muy corto.", "of pretraining, while the supervised CNN starts from scratch with a very short schedule."],
    ["de la CNN con 1.200 etiquetas no representa su capacidad final: su validation accuracy seguía aumentando en la última época.", "for the CNN with 1,200 labels does not represent its final capacity: validation accuracy was still increasing in the last epoch."],
    ["SimCLR supera ampliamente al encoder aleatorio, pero dos épocas no bastan para esperar una representación madura.", "SimCLR clearly beats the random encoder, but two epochs are not enough to expect a mature representation."],
    ["El encoder termina con", "The encoder ends with"],
    [", que comprime la distribución espacial en un vector global.", ", which compresses the spatial distribution into a global vector."],
    ["Logistic Regression conserva los 784 píxeles y sus posiciones.", "Logistic Regression preserves all 784 pixels and their positions."],
    ["Esto puede favorecer al baseline lineal en MNIST, aunque la comparación principal SimCLR–aleatorio sigue siendo justa porque ambos utilizan exactamente el mismo encoder.", "This can favour the linear baseline on MNIST, although the main SimCLR–random comparison remains fair because both use exactly the same encoder."],
    ["En los modelos débiles, F1 macro es menor que accuracy porque las matrices de confusión muestran que algunas clases se predicen mucho mejor que otras.", "For weak models, macro F1 is below accuracy because the confusion matrices show that some classes are predicted much better than others."],
    ["El encoder aleatorio, por ejemplo, alcanza", "The random encoder, for example, reaches"],
    ["de accuracy con todas las etiquetas, pero su F1 macro es mucho menor.", "accuracy with all labels, but its macro F1 is much lower."],
    ["Esto es compatible con un clasificador que se concentra en unas pocas clases; no parece un fallo del cálculo de métricas.", "This is consistent with a classifier focusing on a few classes; it does not look like a metric-computation error."],
    ["En el modo", "In"],
    [", SimCLR aprendió sin etiquetas una representación claramente más útil que la de un encoder aleatorio.", ", SimCLR learned a clearly more useful representation than a random encoder without using labels."],
    ["Sin embargo, no superó a la CNN supervisada ni a Logistic Regression.", "However, it did not outperform the supervised CNN or Logistic Regression."],
    ["Por tanto, esta demo demuestra que el pipeline contrastivo funciona, pero", "Therefore, this demo shows that the contrastive pipeline works, but"],
    ["no demuestra que SimCLR reduzca la necesidad de etiquetas en general", "does not show that SimCLR generally reduces the need for labels"],
    ["No debemos afirmar que", "We should not claim that"],
    ["SimCLR funciona mejor con pocas etiquetas", "SimCLR works better with few labels"],
    [": con 1 % obtiene", ": with 1% it obtains"],
    [", frente a", ", compared with"],
    ["de la CNN y", "for the CNN and"],
    ["de Logistic Regression. Para responder la hipótesis principal con rigor harían falta el modo completo, más preentrenamiento y varias semillas.", "for Logistic Regression. Answering the main hypothesis rigorously would require full mode, more pretraining, and multiple seeds."],
    ["Gracias", "Thank you"],
    ["Preguntas", "Questions"],
    ["Idea clave:", "Key idea:"],
    ["el éxito contrastivo se mide primero por cuánto mejora la representación frente al mismo encoder sin preentrenar, no por asumir que debe ganar siempre a todos los clasificadores.", "contrastive success is first measured by how much the representation improves over the same untrained encoder, not by assuming it must always beat every classifier."],
    ["Entrenamiento", "Training"],
    ["Validación", "Validation"]
  ].sort((a, b) => b[0].length - a[0].length);

  function translate(text) {
    return replacements.reduce((value, [es, en]) => value.split(es).join(en), text);
  }

  function translateNode(node, text) {
    let value = translate(text);
    if (text.trim() === 'y') value = text.replace('y', 'and');
    const cell = node.parentElement && node.parentElement.closest('.jp-Cell');
    if (cell && cell.id === 'cell-id=conclusions') {
      if (text.trim() === 'a') value = text.replace('a', 'to');
      value = value.replace(/\bEl\s*$/, 'The ');
    }
    return value;
  }

  function initialise() {
    const header = document.querySelector("main > .jp-MarkdownCell[id='cell-id=title']");
    const main = document.querySelector("main");
    if (!header || !main || header.querySelector('.language-switcher')) return;

    const switcher = document.createElement('div');
    switcher.className = 'language-switcher';
    switcher.setAttribute('role', 'group');
    switcher.setAttribute('aria-label', 'Language / Idioma');
    switcher.innerHTML = '<button type="button" data-lang="en" aria-pressed="true">EN</button><button type="button" data-lang="es" aria-pressed="false">ES</button>';
    header.appendChild(switcher);

    const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        const parent = node.parentElement;
        if (!parent || parent.closest('style, script, .language-switcher')) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      nodes.push({ node, es: node.nodeValue, en: translateNode(node, node.nodeValue) });
    }

    function setLanguage(language) {
      nodes.forEach(item => { item.node.nodeValue = item[language]; });
      document.documentElement.lang = language;
      document.title = language === 'en'
        ? 'Contrastive Learning with Few Labels'
        : 'Contrastive Learning con pocas etiquetas';
      switcher.querySelectorAll('button').forEach(button => {
        const active = button.dataset.lang === language;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
      });
    }

    switcher.addEventListener('click', event => {
      const button = event.target.closest('button[data-lang]');
      if (button) {
        setLanguage(button.dataset.lang);
        const url = new URL(window.location.href);
        if (button.dataset.lang === 'es') url.searchParams.set('lang', 'es');
        else url.searchParams.delete('lang');
        window.history.replaceState({}, '', url);
      }
    });
    const requestedLanguage = new URLSearchParams(window.location.search).get('lang');
    setLanguage(requestedLanguage === 'es' ? 'es' : 'en');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialise);
  } else {
    initialise();
  }
})();
