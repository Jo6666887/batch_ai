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
import openai  # 添加OpenAI库

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

def process_questions(task_id, questions, system_prompt, api_key, temperature, model_provider="volcano", base_url=None, model_name=None):
    """后台处理问题的函数"""
    logger.info(f"开始处理任务 {task_id}，使用温度参数: {temperature}，模型提供商: {model_provider}")
    results = []
    current_tasks[task_id]['status'] = 'processing'
    current_tasks[task_id]['total'] = len(questions)
    current_tasks[task_id]['completed'] = 0
    current_tasks[task_id]['paused'] = False  # 添加暂停状态标记
    current_tasks[task_id]['last_update'] = time.time()  # 初始化更新时间
    current_tasks[task_id]['current_index'] = 0  # 当前处理的索引
    current_tasks[task_id]['interrupted'] = False  # 是否中断标记
    
    # 使用实际的API密钥
    actual_api_key = api_key if api_key else os.environ.get("ARK_API_KEY")
    
    # 根据提供商初始化客户端
    if model_provider == "volcano":
        # 火山引擎客户端初始化
        actual_base_url = base_url if base_url else "https://ark.cn-beijing.volces.com/api/v3"
        actual_model = model_name if model_name else "ep-20250317192842-2mhtn"
        logger.info(f"初始化火山引擎客户端: base_url={actual_base_url}, model={actual_model}")
        client = Ark(
            base_url=actual_base_url,
            api_key=actual_api_key
        )
    else:
        # OpenAI客户端初始化
        actual_base_url = base_url if base_url else "https://api.openai.com/v1"
        actual_model = model_name if model_name else "gpt-4o"
        logger.info(f"初始化OpenAI客户端: base_url={actual_base_url}, model={actual_model}")
        
        # 打印用于调试的信息，不包含密钥
        logger.info(f"OpenAI配置: URL={actual_base_url}, 模型={actual_model}")
        
        try:
            # 设置HTTP超时，增加兼容性
            import httpx
            # 尝试为OpenAI库创建自定义的HTTP客户端
            http_client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True
            )
            
            # 初始化OpenAI客户端
            client = openai.OpenAI(
                api_key=actual_api_key,
                base_url=actual_base_url,
                http_client=http_client,
                max_retries=3  # 添加重试次数
            )
            logger.info("OpenAI客户端初始化成功")
        except Exception as e:
            # 如果高级配置失败，使用基本配置
            logger.warning(f"使用高级配置初始化OpenAI客户端失败: {str(e)}，尝试使用基本配置")
            client = openai.OpenAI(
                api_key=actual_api_key,
                base_url=actual_base_url
            )
            logger.info("使用基本配置初始化OpenAI客户端成功")
    
    i = 0
    while i < len(questions):
        current_tasks[task_id]['current_index'] = i  # 更新当前索引
        
        # 检查是否暂停
        while current_tasks[task_id].get('paused', False):
            logger.info(f"任务 {task_id} 已暂停，等待继续...")
            # 暂停时每秒检查一次状态
            time.sleep(1)
            # 如果任务被删除，则退出
            if task_id not in current_tasks:
                logger.info(f"任务 {task_id} 已被删除，终止处理")
                return results
            # 当状态改变时立即退出循环
            if not current_tasks[task_id].get('paused', False):
                logger.info(f"任务 {task_id} 已恢复")
                break
            
        question = questions[i]
        
        # 检查是否被标记为中断
        was_interrupted = False
        if len(results) > i and results[i].get('interrupted', False):
            was_interrupted = True
            logger.info(f"重新处理被中断的问题，索引: {i}")
            
        try:
            logger.info(f"处理问题 {i+1}/{len(questions)}，使用温度: {temperature}")
            
            # 如果问题为空，跳过
            if pd.isna(question) or str(question).strip() == '':
                logger.warning(f"跳过空问题: 索引 {i}")
                i += 1  # 移至下一个问题
                continue
                
            # 确保问题是字符串类型
            question = str(question).strip()
            
            # 使用流式API
            answer = ""
            try:
                # 根据不同的模型提供商发送请求
                if model_provider == "volcano":
                    # 火山引擎API
                    logger.info(f"使用火山引擎API调用模型: {actual_model}")
                    stream = client.chat.completions.create(
                        model=actual_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question}
                        ],
                        stream=True,
                        temperature=float(temperature)
                    )
                else:
                    # OpenAI API
                    logger.info(f"使用OpenAI API调用模型: {actual_model}, 基础URL: {actual_base_url}")
                    try:
                        stream = client.chat.completions.create(
                            model=actual_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": question}
                            ],
                            stream=True,
                            temperature=float(temperature)
                        )
                        logger.info("OpenAI API请求成功发送")
                    except Exception as e:
                        logger.error(f"OpenAI API请求失败: {str(e)}")
                        raise
                
                # 实时收集流式响应
                for chunk in stream:
                    # 检查是否在收集响应过程中被暂停
                    if current_tasks[task_id].get('paused', False):
                        logger.info(f"任务 {task_id} 在流式响应过程中被暂停")
                        # 记录部分结果
                        if answer:
                            result = {
                                "question": question,
                                "answer": answer + "\n\n[由于任务暂停，回答未完成]",
                                "is_streaming": False,
                                "interrupted": True
                            }
                            
                            if len(results) > i:
                                results[i] = result
                            else:
                                results.append(result)
                            
                            current_tasks[task_id]['results'] = results.copy()
                            current_tasks[task_id]['last_update'] = time.time()
                            current_tasks[task_id]['interrupted'] = True
                            save_results(task_id, results)
                        
                        # 停止处理当前流
                        break
                    
                    # 根据不同API处理响应流
                    if model_provider == "volcano":
                        if not chunk.choices:
                            continue
                        if chunk.choices[0].delta and chunk.choices[0].delta.content:
                            answer += chunk.choices[0].delta.content
                    else:
                        # OpenAI API处理方式
                        try:
                            if not chunk.choices:
                                logger.debug("OpenAI响应chunk没有choices")
                                continue
                            
                            # 记录OpenAI响应的结构
                            if i == 0 and len(answer) < 50:
                                logger.info(f"OpenAI响应chunk结构: {chunk}")
                            
                            # 检查delta结构，不同版本的OpenAI库可能有不同的结构
                            delta = chunk.choices[0].delta
                            
                            # 尝试多种方式获取内容
                            content = None
                            # 方法1: 直接获取content属性
                            if hasattr(delta, 'content') and delta.content is not None:
                                content = delta.content
                            # 方法2: 尝试作为字典访问
                            elif hasattr(delta, 'get'):
                                content = delta.get('content')
                            # 方法3: 尝试获取字典格式
                            elif isinstance(delta, dict):
                                content = delta.get('content')
                            
                            if content:
                                answer += content
                                if len(answer) < 50:  # 只记录前50个字符以避免日志过大
                                    logger.debug(f"已接收内容: {answer}")
                        except Exception as e:
                            logger.error(f"处理OpenAI响应流错误: {str(e)}，chunk类型: {type(chunk)}")
                            # 尝试直接获取JSON数据
                            try:
                                logger.info(f"尝试直接解析chunk: {chunk}")
                                if hasattr(chunk, 'model_dump_json'):
                                    logger.info(f"chunk model_dump_json: {chunk.model_dump_json()}")
                            except Exception as parse_err:
                                logger.error(f"解析chunk失败: {str(parse_err)}")
                            continue
                    
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
                
                # 检查是否因为暂停而中断了流式处理
                if current_tasks[task_id].get('paused', False):
                    # 不增加索引i，这样恢复时会重新处理当前问题
                    continue
                
                # 流式输出完成后，移除streaming标记
                if len(results) > i:
                    results[i]["is_streaming"] = False
                    # 如果这是一个被中断重新处理的问题，移除interrupted标记
                    if was_interrupted:
                        results[i].pop('interrupted', None)
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
            i += 1  # 只有成功处理后才增加索引
            current_tasks[task_id]['completed'] = i
            current_tasks[task_id]['results'] = results
            
            # 保存中间结果
            try:
                save_results(task_id, results)
                logger.info(f"已保存中间结果，完成度: {i}/{len(questions)}")
            except Exception as e:
                logger.error(f"保存结果出错: {str(e)}")
            
            # 添加短暂延迟，避免请求过于频繁
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            i += 1  # 错误时也要移至下一个问题
            continue
    
    current_tasks[task_id]['status'] = 'completed'
    current_tasks[task_id]['interrupted'] = False
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
    # 检查是否有保存的任务状态
    last_task_id = request.args.get('last_task_id')
    if last_task_id and last_task_id in current_tasks:
        task = current_tasks[last_task_id]
        system_prompt = task.get('system_prompt', '')
        temperature = task.get('temperature', 0.7)
        model_provider = task.get('model_provider', 'volcano')
        volcano_base_url = task.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3') if model_provider == 'volcano' else 'https://ark.cn-beijing.volces.com/api/v3'
        volcano_model = task.get('model_name', 'ep-20250317192842-2mhtn') if model_provider == 'volcano' else 'ep-20250317192842-2mhtn'
        openai_base_url = task.get('base_url', 'https://api.openai.com/v1') if model_provider == 'openai' else 'https://api.openai.com/v1'
        openai_model = task.get('model_name', 'gpt-4o') if model_provider == 'openai' else 'gpt-4o'
        
        return render_template('index.html', 
                              last_task_id=last_task_id,
                              system_prompt=system_prompt,
                              temperature=temperature,
                              model_provider=model_provider,
                              volcano_base_url=volcano_base_url,
                              volcano_model=volcano_model,
                              openai_base_url=openai_base_url,
                              openai_model=openai_model)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    system_prompt = request.form.get('system_prompt', '你是人工智能助手。')
    api_key = request.form.get('api_key', os.getenv('ARK_API_KEY', ''))
    temperature = float(request.form.get('temperature', 0.7))
    
    # 获取模型提供商和相关设置
    model_provider = request.form.get('model_provider', 'volcano')
    
    # 根据提供商获取不同的设置
    if model_provider == 'volcano':
        base_url = request.form.get('volcano_base_url', 'https://ark.cn-beijing.volces.com/api/v3')
        model_name = request.form.get('volcano_model', 'ep-20250317192842-2mhtn')
    else:
        base_url = request.form.get('openai_base_url', 'https://api.openai.com/v1')
        model_name = request.form.get('openai_model', 'gpt-4o')
    
    logger.info(f"收到请求：提供商={model_provider}, 模型={model_name}, 温度={temperature}")
    
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
                'original_file_path': filepath,  # 保存原始文件路径
                'system_prompt': system_prompt,  # 保存系统提示词
                'api_key': api_key,  # 保存API密钥
                'temperature': temperature,  # 保存温度参数
                'model_provider': model_provider,  # 保存模型提供商
                'base_url': base_url,  # 保存基础URL
                'model_name': model_name,  # 保存模型名称
                'status': 'starting',
                'total': len(questions),
                'completed': 0,
                'results': []
            }
            
            thread = threading.Thread(
                target=process_questions,
                args=(task_id, questions, system_prompt, api_key, temperature, model_provider, base_url, model_name)
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
                'results': results,
                'paused': False,
                'last_update': time.time(),
                'interrupted': False,
                'current_index': len(results)
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
                'results': results,
                'paused': False,
                'interrupted': False,
                'current_index': -1,
                'last_update': time.time()
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
        'results': task['results'],
        'last_update': task.get('last_update', time.time()),
        'paused': task.get('paused', False),
        'interrupted': task.get('interrupted', False),
        'current_index': task.get('current_index', -1)
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

@app.route('/api/task/<task_id>/control', methods=['POST'])
def control_task(task_id):
    """控制任务的暂停和继续"""
    if task_id not in current_tasks:
        return jsonify({'error': '任务不存在'}), 404
    
    data = request.get_json()
    action = data.get('action')
    
    if action == 'pause':
        current_tasks[task_id]['paused'] = True
        logger.info(f"任务 {task_id} 已暂停")
        return jsonify({
            'status': 'paused',
            'task_info': {
                'system_prompt': current_tasks[task_id].get('system_prompt', ''),
                'filename': current_tasks[task_id].get('filename', ''),
                'temperature': current_tasks[task_id].get('temperature', 0.7),
                'model_provider': current_tasks[task_id].get('model_provider', 'volcano'),
                'base_url': current_tasks[task_id].get('base_url', ''),
                'model_name': current_tasks[task_id].get('model_name', '')
            }
        })
    elif action == 'resume':
        current_tasks[task_id]['paused'] = False
        logger.info(f"任务 {task_id} 已继续")
        return jsonify({'status': 'resumed'})
    else:
        return jsonify({'error': '无效的操作'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)