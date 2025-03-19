import os
import json
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
import threading
import time
from dotenv import load_dotenv
import logging
from volcenginesdkarkruntime import Ark

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

app = Flask(__name__, 
            template_folder='app/templates')
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
app.config['RESULTS_FOLDER'] = 'results'

# 确保上传和结果目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# 全局变量用于存储当前正在处理的任务
current_tasks = {}

def process_questions(task_id, questions, system_prompt, api_key):
    """后台处理问题的函数"""
    results = []
    current_tasks[task_id]['status'] = 'processing'
    current_tasks[task_id]['total'] = len(questions)
    current_tasks[task_id]['completed'] = 0
    
    # 使用实际的API密钥
    actual_api_key = api_key if api_key else os.environ.get("ARK_API_KEY")
    
    # 初始化Ark客户端
    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=actual_api_key
    )
    
    for i, question in enumerate(questions):
        try:
            logger.info(f"处理问题 {i+1}/{len(questions)}")
            
            # 如果问题为空，跳过
            if pd.isna(question) or str(question).strip() == '':
                logger.warning(f"跳过空问题: 索引 {i}")
                continue
                
            # 确保问题是字符串类型
            question = str(question).strip()
            
            # 使用流式API
            answer = ""
            try:
                # 发送流式请求
                stream = client.chat.completions.create(
                    model="ep-20250317192842-2mhtn",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    stream=True,
                    temperature=0.6
                )
                
                # 实时收集流式响应
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    if chunk.choices[0].delta and chunk.choices[0].delta.content:
                        answer += chunk.choices[0].delta.content
                        
                        # 每次有新内容时更新当前任务状态
                        result = {
                            "question": question,
                            "answer": answer,
                            "is_streaming": True
                        }
                        
                        # 更新当前结果
                        if len(results) > i:
                            results[i] = result
                        else:
                            results.append(result)
                            
                        # 更新任务状态
                        current_tasks[task_id]['results'] = results.copy()
                        current_tasks[task_id]['last_update'] = time.time()
                        
                        # 每积累一定字符保存一次中间结果
                        if len(answer) % 50 == 0:
                            try:
                                save_results(task_id, results)
                            except Exception as e:
                                logger.error(f"保存中间结果出错: {str(e)}")
                
                # 流式输出完成后，移除streaming标记
                if len(results) > i:
                    results[i]["is_streaming"] = False
                logger.info(f"成功获取流式回答: {answer[:100]}...")
                
            except Exception as e:
                error_msg = f"处理出错: {str(e)}"
                logger.error(error_msg)
                answer = error_msg
                
                # 记录错误结果
                result = {
                    "question": question,
                    "answer": answer,
                    "is_streaming": False,
                    "error": True
                }
                
                if len(results) > i:
                    results[i] = result
                else:
                    results.append(result)
            
            # 更新进度
            current_tasks[task_id]['completed'] = i + 1
            current_tasks[task_id]['results'] = results
            
            # 保存中间结果
            try:
                save_results(task_id, results)
                logger.info(f"已保存中间结果，完成度: {i+1}/{len(questions)}")
            except Exception as e:
                logger.error(f"保存结果出错: {str(e)}")
            
            # 添加短暂延迟，避免请求过于频繁
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            continue
    
    current_tasks[task_id]['status'] = 'completed'
    save_results(task_id, results)
    logger.info(f"任务 {task_id} 完成，共处理了 {len(results)} 个问题")
    return results

def save_results(task_id, results):
    """保存结果到文件"""
    filename = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    system_prompt = request.form.get('system_prompt', '你是人工智能助手。')
    api_key = request.form.get('api_key', os.getenv('ARK_API_KEY', ''))
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # 生成任务ID
        task_id = str(int(time.time()))
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(filepath)
        
        # 读取Excel文件
        try:
            logger.info(f"读取Excel文件: {filepath}")
            df = pd.read_excel(filepath)
            logger.info(f"Excel文件形状: {df.shape}")
            
            # 获取所有非空问题
            questions = []
            for i, value in enumerate(df.iloc[:, 0]):
                if pd.notna(value) and str(value).strip():
                    questions.append(str(value).strip())
                    logger.info(f"添加问题{i+1}: {str(value).strip()[:30]}...")
            
            logger.info(f"共找到 {len(questions)} 个有效问题")
            
            # 启动后台处理线程
            current_tasks[task_id] = {
                'filename': filename,
                'status': 'starting',
                'total': len(questions),
                'completed': 0,
                'results': []
            }
            
            thread = threading.Thread(
                target=process_questions,
                args=(task_id, questions, system_prompt, api_key)
            )
            thread.daemon = True
            thread.start()
            
            return redirect(url_for('task_status', task_id=task_id))
        
        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"处理文件时出错: {str(e)}", 500

@app.route('/task/<task_id>')
def task_status(task_id):
    if task_id not in current_tasks:
        # 尝试从保存的文件中恢复任务状态
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            current_tasks[task_id] = {
                'filename': 'recovered_task',
                'status': 'completed',
                'total': len(results),
                'completed': len(results),
                'results': results
            }
            logger.info(f"从文件恢复任务 {task_id}，共 {len(results)} 个结果")
        else:
            logger.warning(f"未找到任务 {task_id}")
            return "未找到任务", 404
    
    return render_template('task.html', task_id=task_id)

@app.route('/api/task/<task_id>')
def get_task_status(task_id):
    if task_id not in current_tasks:
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"API请求：任务 {task_id} 从文件加载，包含 {len(results)} 个结果")
            return jsonify({
                'status': 'completed',
                'total': len(results),
                'completed': len(results),
                'results': results
            })
        logger.warning(f"API请求：未找到任务 {task_id}")
        return jsonify({'error': '未找到任务'}), 404
    
    task = current_tasks[task_id]
    logger.info(f"API请求：获取任务 {task_id} 状态，{task['completed']}/{task['total']}")
    return jsonify({
        'status': task['status'],
        'filename': task['filename'],
        'total': task['total'],
        'completed': task['completed'],
        'results': task['results']
    })

@app.route('/download/<task_id>')
def download_results(task_id):
    from flask import Response
    
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
    if not os.path.exists(result_file):
        logger.warning(f"下载请求：未找到任务 {task_id} 的结果文件")
        return "结果文件不存在", 404
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    logger.info(f"下载请求：生成任务 {task_id} 的CSV文件，包含 {len(results)} 个结果")
    
    # 创建CSV格式的响应
    csv_data = "问题,回答\n"
    for item in results:
        q = item['question'].replace('"', '""')
        a = item['answer'].replace('"', '""')
        csv_data += f'"{q}","{a}"\n'
    
    # 返回下载响应
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=results_{task_id}.csv"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)