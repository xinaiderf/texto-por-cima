from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import requests
import tempfile
import cv2
import os
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()

# Função para converter cor hexadecimal para RGB (usado pelo Pillow)
def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

# Modelo para o request com parâmetros de personalização do texto
class VideoPrompt(BaseModel):
    video_url: str
    prompt: str
    pos_x: int = None             # Posição horizontal do texto (pixels)
    pos_y: int = None             # Posição vertical do texto (pixels)
    font_scale: float = 1.0       # Fator de escala para o tamanho da fonte
    thickness: int = 2            # Espessura do texto (usada como stroke_width)
    text_color: str = "#ffffff"   # Cor do texto em hexadecimal
    font: str = "arial.ttf"       # Nome (ou caminho) do arquivo da fonte TrueType
    max_chars: int = None         # Máximo de caracteres por linha

@app.post("/overlay")
async def overlay_text(data: VideoPrompt):
    # Baixa o vídeo a partir da URL informada
    try:
        response = requests.get(data.video_url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Não foi possível baixar o vídeo")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Salva o vídeo em um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                temp_video_file.write(chunk)
        temp_video_file.flush()
        video_path = temp_video_file.name

    # Carrega o vídeo com OpenCV (somente para leitura dos frames)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Não foi possível abrir o vídeo")
    
    # Obtém as propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Cria um arquivo temporário para o vídeo de saída
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output_file:
        output_path = temp_output_file.name
    
    # Inicializa o VideoWriter para salvar o vídeo final
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Prepara a quebra de linha do texto, se max_chars for definido
    if data.max_chars is not None:
        linhas = textwrap.wrap(data.prompt, width=data.max_chars)
    else:
        linhas = [data.prompt]
    
    # Espaçamento entre linhas (em pixels)
    line_gap = 10
    
    # Configuração da fonte usando PIL (toda a parte de texto será processada pelo Pillow)
    font_size = int(data.font_scale * 20)
    try:
        fonte_pil = ImageFont.truetype(data.font, font_size)
    except IOError:
        fonte_pil = ImageFont.load_default()
    
    # Calcula a altura da linha usando os métricos da fonte (Pillow)
    ascent, descent = fonte_pil.getmetrics()
    line_height = ascent + descent
    
    # Se pos_y não for definida, posiciona o bloco de texto acima da borda inferior com uma margem de 50 pixels
    if data.pos_y is None:
        total_text_height = len(linhas) * line_height + (len(linhas) - 1) * line_gap
        base_y = height - total_text_height - 50
    else:
        base_y = data.pos_y
    
    # Converte a cor do texto (hexadecimal) para RGB
    color = hex_to_rgb(data.text_color)
    
    # Processa cada frame do vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converte o frame (BGR do OpenCV) para uma imagem PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(image_pil)
        
        # Para cada linha de texto, calcula a largura e posiciona
        for i, linha in enumerate(linhas):
            bbox = draw.textbbox((0, 0), linha, font=fonte_pil)
            line_width = bbox[2] - bbox[0]
            x = data.pos_x if data.pos_x is not None else (width - line_width) // 2
            y = base_y + i * (line_height + line_gap)
            # Desenha o texto com stroke (contorno) para simular a espessura
            draw.text((x, y), linha, fill=color, font=fonte_pil,
                      stroke_width=data.thickness, stroke_fill="Black")
        
        # Converte a imagem PIL de volta para o formato BGR do OpenCV
        frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    # Libera os recursos e remove o arquivo temporário do vídeo original
    cap.release()
    out.release()
    os.remove(video_path)
    
    return FileResponse(output_path, media_type='video/mp4', filename="output.mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)
