import os
import json
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
import threading
import time
import concurrent.futures
from dotenv import load_dotenv
import logging
from volcenginesdkarkruntime import Ark
import openai  # 添加OpenAI库
import httpx
import asyncio
from functools import partial
import re

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

# 创建线程池
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# 创建异步HTTP客户端
async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=5.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    http2=True  # 已安装h2依赖，启用HTTP/2以提高性能
)

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
        
        # 创建多个客户端实例以提高并发性能
        clients = [
            Ark(
                base_url=actual_base_url,
                api_key=actual_api_key
            ) for _ in range(min(5, len(questions)))  # 最多创建5个客户端
        ]
        
        # 如果没有问题，至少创建一个客户端
        if not clients:
            clients = [Ark(
                base_url=actual_base_url,
                api_key=actual_api_key
            )]
            
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
            # 创建多个客户端以提高并发性能
            http_clients = [
                httpx.Client(
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    follow_redirects=True
                ) for _ in range(min(5, len(questions)))
            ]
            
            # 初始化多个OpenAI客户端
            clients = [
                openai.OpenAI(
                    api_key=actual_api_key,
                    base_url=actual_base_url,
                    http_client=http_client,
                    max_retries=3
                ) for http_client in http_clients
            ]
            logger.info("OpenAI客户端初始化成功")
        except Exception as e:
            # 如果高级配置失败，使用基本配置
            logger.warning(f"使用高级配置初始化OpenAI客户端失败: {str(e)}，尝试使用基本配置")
            clients = [openai.OpenAI(
                api_key=actual_api_key,
                base_url=actual_base_url
            )]
            logger.info("使用基本配置初始化OpenAI客户端成功")
    
    # 批量处理函数
    def process_batch(batch_questions, start_idx):
        batch_results = []
        for idx, question in enumerate(batch_questions):
            i = start_idx + idx
            current_tasks[task_id]['current_index'] = i
            
            # 检查是否暂停
            while current_tasks[task_id].get('paused', False):
                logger.info(f"任务 {task_id} 已暂停，等待继续...")
                time.sleep(1)
                if task_id not in current_tasks:
                    logger.info(f"任务 {task_id} 已被删除，终止处理")
                    return []
                if not current_tasks[task_id].get('paused', False):
                    logger.info(f"任务 {task_id} 已恢复")
                    break
            
            # 跳过空问题
            if pd.isna(question) or str(question).strip() == '':
                logger.warning(f"跳过空问题: 索引 {i}")
                batch_results.append(None)
                continue
            
            # 确保问题是字符串类型
            question = str(question).strip()
            
            # 选择客户端（轮询方式）
            client_idx = i % len(clients)
            client = clients[client_idx]
            
            # 检查是否被标记为中断
            was_interrupted = False
            if len(results) > i and results[i].get('interrupted', False):
                was_interrupted = True
                logger.info(f"重新处理被中断的问题，索引: {i}")
            
            try:
                logger.info(f"处理问题 {i+1}/{len(questions)}，使用温度: {temperature}")
                
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
                                
                                while len(batch_results) <= idx:
                                    batch_results.append(None)
                                batch_results[idx] = result
                                
                                # 更新任务状态
                                current_tasks[task_id]['last_update'] = time.time()
                                current_tasks[task_id]['interrupted'] = True
                            
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
                        
                        # 每次有新内容时更新结果
                        result = {
                            "question": question,
                            "answer": answer,
                            "is_streaming": True
                        }
                        
                        # 更新当前结果
                        while len(batch_results) <= idx:
                            batch_results.append(None)
                        batch_results[idx] = result
                    
                    # 检查是否因为暂停而中断了流式处理
                    if current_tasks[task_id].get('paused', False):
                        # 不增加索引i，这样恢复时会重新处理当前问题
                        continue
                    
                    # 流式输出完成后，移除streaming标记
                    result = {
                        "question": question,
                        "answer": answer,
                        "is_streaming": False
                    }
                    # 如果这是一个被中断重新处理的问题，移除interrupted标记
                    if was_interrupted:
                        result.pop('interrupted', None)
                    
                    while len(batch_results) <= idx:
                        batch_results.append(None)
                    batch_results[idx] = result
                    
                    logger.info(f"成功获取流式回答: {answer[:100]}...")
                    
                except Exception as e:
                    error_msg = f"API请求或流处理过程中出错: {str(e)}"
                    logger.error(error_msg)
                    result = {
                        "question": question,
                        "answer": f"处理出错: {error_msg}",
                        "is_streaming": False,
                        "error": True
                    }
                    while len(batch_results) <= idx:
                        batch_results.append(None)
                    batch_results[idx] = result
            
            except Exception as e:
                error_msg = f"处理问题时出错: {str(e)}"
                logger.error(error_msg)
                result = {
                    "question": question,
                    "answer": f"处理出错: {error_msg}",
                    "is_streaming": False,
                    "error": True
                }
                while len(batch_results) <= idx:
                    batch_results.append(None)
                batch_results[idx] = result
        
        return batch_results
    
    # 将问题分成较小的批次并行处理
    batch_size = 3  # 每批最多处理3个问题
    i = 0
    while i < len(questions):
        # 检查任务是否暂停或删除
        if task_id not in current_tasks or current_tasks[task_id].get('paused', False):
            if task_id not in current_tasks:
                logger.info(f"任务 {task_id} 已被删除，终止处理")
                return results
            # 如果暂停，等待继续
            while current_tasks[task_id].get('paused', False):
                time.sleep(1)
                # 如果任务被删除，退出
                if task_id not in current_tasks:
                    logger.info(f"任务 {task_id} 已被删除，终止处理")
                    return results
            
        # 获取当前批次
        batch_end = min(i + batch_size, len(questions))
        batch = questions[i:batch_end]
        
        # 使用线程池处理当前批次
        future = executor.submit(process_batch, batch, i)
        batch_results = future.result()
        
        # 更新结果
        for j, result in enumerate(batch_results):
            if result is not None:
                idx = i + j
                if len(results) <= idx:
                    # 扩展结果列表以适应当前索引
                    results.extend([None] * (idx - len(results) + 1))
                results[idx] = result
                
                # 更新任务状态
                current_tasks[task_id]['results'] = results.copy()
                
                # 每积累一定字符保存一次中间结果
                if idx % 3 == 0 or idx == len(questions) - 1:
                    try:
                        save_results(task_id, results)
                    except Exception as e:
                        logger.error(f"保存中间结果出错: {str(e)}")
        
        # 更新索引和完成数量
        i = batch_end
        current_tasks[task_id]['completed'] = i
        current_tasks[task_id]['last_update'] = time.time()
    
    # 清理非流式结果中的is_streaming标记
    for result in results:
        if result and 'is_streaming' in result:
            result.pop('is_streaming', None)
    
    # 保存最终结果
    current_tasks[task_id]['status'] = 'completed'
    current_tasks[task_id]['completed'] = len(questions)
    current_tasks[task_id]['last_update'] = time.time()
    save_results(task_id, results)
    
    return results

def save_results(task_id, results):
    """保存结果到文件"""
    filename = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存任务元数据（包括系统提示词）
    if task_id in current_tasks:
        metadata_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}_metadata.json')
        metadata = {
            'system_prompt': current_tasks[task_id].get('system_prompt', ''),
            'temperature': current_tasks[task_id].get('temperature', 0.7),
            'model_provider': current_tasks[task_id].get('model_provider', 'volcano'),
            'model_name': current_tasks[task_id].get('model_name', ''),
            'timestamp': time.time()
        }
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存元数据出错: {str(e)}")

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
    
    # 生成任务ID
    task_id = str(int(time.time()))
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(filepath)
        
        # 读取文件 - 支持Excel文件或JSON格式的生成问题
        try:
            logger.info(f"读取文件: {filepath}")
            
            questions = []
            
            # 根据文件类型处理
            if filename.endswith(('.xlsx', '.xls')):
                # 处理Excel文件
                df = pd.read_excel(filepath)
                logger.info(f"Excel文件形状: {df.shape}")
                
                # 获取所有非空问题
                for i, value in enumerate(df.iloc[:, 0]):
                    if pd.notna(value) and str(value).strip():
                        questions.append(str(value).strip())
                        logger.info(f"添加问题{i+1}: {str(value).strip()[:30]}...")
            
            elif filename.endswith(('.json', '.jsonl')):
                # 处理JSON文件（从生成问题保存的）
                with open(filepath, 'r', encoding='utf-8') as f:
                    if filename.endswith('.jsonl'):
                        # 处理JSONL格式
                        questions = []
                        for line in f:
                            item = json.loads(line.strip())
                            if isinstance(item, str):
                                questions.append(item)
                            elif isinstance(item, dict) and 'question' in item:
                                questions.append(item['question'])
                    else:
                        # 处理JSON格式
                        data = json.load(f)
                        if isinstance(data, list):
                            if all(isinstance(item, str) for item in data):
                                # 直接使用字符串列表
                                questions = data
                            elif all(isinstance(item, dict) for item in data) and 'question' in data[0]:
                                # 使用对象列表，提取question字段
                                questions = [item['question'] for item in data if 'question' in item]
                
                logger.info(f"从JSON文件加载了{len(questions)}个问题")
            
            else:
                logger.error(f"不支持的文件类型: {filename}")
                return "不支持的文件类型，请上传Excel或JSON文件", 400
            
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
            
            # 尝试获取任务的元数据
            metadata_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}_metadata.json')
            system_prompt = ""
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        system_prompt = metadata.get('system_prompt', '')
                except Exception as e:
                    logger.error(f"读取元数据文件出错: {str(e)}")
            
            current_tasks[task_id] = {
                'filename': 'recovered_task',
                'status': 'completed',
                'total': len(results),
                'completed': len(results),
                'results': results,
                'paused': False,
                'last_update': time.time(),
                'interrupted': False,
                'current_index': len(results),
                'system_prompt': system_prompt
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
            
            # 尝试获取系统提示词
            system_prompt = ""
            metadata_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}_metadata.json')
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        system_prompt = metadata.get('system_prompt', '')
                except Exception as e:
                    logger.error(f"读取元数据文件出错: {str(e)}")
            
            logger.info(f"API请求：任务 {task_id} 从文件加载，包含 {len(results)} 个结果")
            return jsonify({
                'status': 'completed',
                'total': len(results),
                'completed': len(results),
                'results': results,
                'paused': False,
                'interrupted': False,
                'current_index': -1,
                'last_update': time.time(),
                'system_prompt': system_prompt
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
        'current_index': task.get('current_index', -1),
        'system_prompt': task.get('system_prompt', '')
    })

@app.route('/download/<task_id>')
def download_results(task_id):
    from flask import Response, request
    
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
    if not os.path.exists(result_file):
        logger.warning(f"下载请求：未找到任务 {task_id} 的结果文件")
        return "结果文件不存在", 404
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 获取请求的格式
    format_type = request.args.get('format', 'csv')
    logger.info(f"下载请求：生成任务 {task_id} 的{format_type}文件，包含 {len(results)} 个结果")
    
    # 获取系统提示词作为instruction
    system_prompt = ""
    # 首先尝试从内存中的任务获取
    if task_id in current_tasks:
        system_prompt = current_tasks[task_id].get('system_prompt', '')
    
    # 如果内存中没有，尝试从元数据文件获取
    if not system_prompt:
        metadata_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}_metadata.json')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    system_prompt = metadata.get('system_prompt', '')
                    logger.info(f"从元数据文件加载系统提示词: {system_prompt[:30]}...")
            except Exception as e:
                logger.error(f"读取元数据文件出错: {str(e)}")
    
    # 如果仍然没有，使用默认值
    if not system_prompt:
        system_prompt = "你是人工智能助手。"
        logger.info("使用默认系统提示词")
    
    if format_type == 'csv':
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
    elif format_type == 'json':
        # 创建Alpaca格式的JSON，系统提示词为instruction，问题为input
        alpaca_data = []
        for item in results:
            alpaca_item = {
                "instruction": system_prompt,
                "input": item['question'],
                "output": item['answer'],
                "system": "",
                "history": ""
            }
            alpaca_data.append(alpaca_item)
        
        # 返回JSON格式
        return Response(
            json.dumps(alpaca_data, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-disposition": f"attachment; filename=alpaca_results_{task_id}.json"}
        )
    elif format_type == 'jsonl':
        # 创建Alpaca格式的JSONL，系统提示词为instruction，问题为input
        jsonl_data = ""
        for item in results:
            alpaca_item = {
                "instruction": system_prompt,
                "input": item['question'],
                "output": item['answer'],
                "system": "",
                "history": ""
            }
            jsonl_data += json.dumps(alpaca_item, ensure_ascii=False) + "\n"
        
        # 返回JSONL格式
        return Response(
            jsonl_data,
            mimetype="application/jsonl",
            headers={"Content-disposition": f"attachment; filename=alpaca_results_{task_id}.jsonl"}
        )
    else:
        return "不支持的格式类型", 400

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

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    """生成问题数据集"""
    generation_prompt = request.form.get('generation_prompt', '')
    num_questions = int(request.form.get('num_questions', 10))
    gen_system_prompt = request.form.get('gen_system_prompt', '你是专业的问题生成器，善于根据主题创建有深度、多样化的问题。')
    gen_temperature = float(request.form.get('gen_temperature', 0.8))
    api_key = request.form.get('api_key', os.getenv('ARK_API_KEY', ''))
    
    # 限制生成数量，避免过大请求
    num_questions = min(max(1, num_questions), 100)
    
    # 使用默认模型提供商
    model_provider = "volcano"
    base_url = "https://ark.cn-beijing.volces.com/api/v3"
    model_name = "ep-20250317192842-2mhtn"
    
    # 创建生成任务
    task_id = f"gen_{int(time.time())}"
    
    # 构建提示词
    prompt = f"""根据下面的要求生成{num_questions}个不同的问题。每个问题应该独特、清晰且富有价值。
    
要求：{generation_prompt}

请按以下格式返回问题列表：
1. 第一个问题
2. 第二个问题
...

不要包含答案或解释，只需生成问题列表。"""
    
    # 创建任务状态
    current_tasks[task_id] = {
        'status': 'starting',
        'type': 'question_generation',  # 标记此任务为问题生成任务
        'generation_prompt': generation_prompt,
        'num_questions': num_questions,
        'gen_system_prompt': gen_system_prompt,
        'gen_temperature': gen_temperature,
        'model_provider': model_provider,
        'base_url': base_url,
        'model_name': model_name,
        'api_key': api_key,
        'total': num_questions,
        'completed': 0,
        'results': [],
        'paused': False,
        'last_update': time.time()
    }
    
    # 启动后台处理线程
    thread = threading.Thread(
        target=process_question_generation,
        args=(task_id, prompt, gen_system_prompt, api_key, gen_temperature, model_provider, base_url, model_name)
    )
    thread.daemon = True
    thread.start()
    
    return redirect(url_for('generation_status', task_id=task_id))

@app.route('/generation/<task_id>')
def generation_status(task_id):
    """展示问题生成任务的状态和结果"""
    if task_id not in current_tasks or current_tasks[task_id].get('type') != 'question_generation':
        # 尝试从保存的文件中恢复任务状态
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 尝试获取任务的元数据
            metadata_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}_metadata.json')
            gen_system_prompt = ""
            gen_temperature = 0.8
            generation_prompt = ""
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        gen_system_prompt = metadata.get('gen_system_prompt', '')
                        gen_temperature = metadata.get('gen_temperature', 0.8)
                        generation_prompt = metadata.get('generation_prompt', '')
                except Exception as e:
                    logger.error(f"读取元数据文件出错: {str(e)}")
            
            current_tasks[task_id] = {
                'status': 'completed',
                'type': 'question_generation',
                'total': len(results),
                'completed': len(results),
                'results': results,
                'paused': False,
                'last_update': time.time(),
                'gen_system_prompt': gen_system_prompt,
                'gen_temperature': gen_temperature,
                'generation_prompt': generation_prompt
            }
            logger.info(f"从文件恢复生成任务 {task_id}，共 {len(results)} 个结果")
        else:
            logger.warning(f"未找到生成任务 {task_id}")
            return "未找到生成任务", 404
    
    return render_template('generation.html', task_id=task_id)

@app.route('/api/generation/<task_id>')
def get_generation_status(task_id):
    """获取问题生成任务的状态和结果"""
    if task_id not in current_tasks or current_tasks[task_id].get('type') != 'question_generation':
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 尝试获取元数据
            metadata_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}_metadata.json')
            gen_system_prompt = ""
            gen_temperature = 0.8
            generation_prompt = ""
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        gen_system_prompt = metadata.get('gen_system_prompt', '')
                        gen_temperature = metadata.get('gen_temperature', 0.8)
                        generation_prompt = metadata.get('generation_prompt', '')
                except Exception as e:
                    logger.error(f"读取元数据文件出错: {str(e)}")
            
            logger.info(f"API请求：生成任务 {task_id} 从文件加载，包含 {len(results)} 个结果")
            return jsonify({
                'status': 'completed',
                'total': len(results),
                'completed': len(results),
                'results': results,
                'generation_prompt': generation_prompt,
                'gen_system_prompt': gen_system_prompt,
                'gen_temperature': gen_temperature,
                'last_update': time.time(),
                'is_streaming': False
            })
        logger.warning(f"API请求：未找到生成任务 {task_id}")
        return jsonify({'error': '未找到生成任务'}), 404
    
    task = current_tasks[task_id]
    logger.info(f"API请求：获取生成任务 {task_id} 状态，{task['completed']}/{task['total']}")
    return jsonify({
        'status': task['status'],
        'total': task['total'],
        'completed': task['completed'],
        'results': task['results'],
        'generation_prompt': task.get('generation_prompt', ''),
        'gen_system_prompt': task.get('gen_system_prompt', ''),
        'gen_temperature': task.get('gen_temperature', 0.8),
        'last_update': task.get('last_update', time.time()),
        'is_streaming': task.get('is_streaming', False),
        'paused': task.get('paused', False)
    })

@app.route('/download/generation/<task_id>')
def download_generated_questions(task_id):
    """下载生成的问题"""
    from flask import Response, request
    
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
    if not os.path.exists(result_file):
        logger.warning(f"下载请求：未找到生成任务 {task_id} 的结果文件")
        return "结果文件不存在", 404
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 获取请求的格式
    format_type = request.args.get('format', 'excel')
    logger.info(f"下载请求：生成任务 {task_id} 的{format_type}文件，包含 {len(results)} 个结果")
    
    if format_type == 'excel':
        # 创建Excel文件
        import io
        import pandas as pd
        from openpyxl import Workbook
        
        output = io.BytesIO()
        df = pd.DataFrame(results, columns=['question'])
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='生成的问题')
        
        output.seek(0)
        
        # 返回Excel响应
        return Response(
            output.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-disposition": f"attachment; filename=generated_questions_{task_id}.xlsx"}
        )
    elif format_type == 'text':
        # 创建文本格式
        text_data = "\n".join([f"{i+1}. {q}" for i, q in enumerate(results)])
        
        # 返回文本响应
        return Response(
            text_data,
            mimetype="text/plain",
            headers={"Content-disposition": f"attachment; filename=generated_questions_{task_id}.txt"}
        )
    else:
        return "不支持的格式类型", 400

@app.route('/api/generation/<task_id>/pause', methods=['POST'])
def pause_generation(task_id):
    """暂停问题生成任务"""
    if task_id in current_tasks:
        current_tasks[task_id]['paused'] = True
        current_tasks[task_id]['last_update'] = time.time()
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '任务不存在'}), 404

@app.route('/api/generation/<task_id>/resume', methods=['POST'])
def resume_generation(task_id):
    """继续问题生成任务"""
    if task_id in current_tasks:
        current_tasks[task_id]['paused'] = False
        current_tasks[task_id]['last_update'] = time.time()
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '任务不存在'}), 404

def process_question_generation(task_id, prompt, system_prompt, api_key, temperature, model_provider="volcano", base_url=None, model_name=None):
    """后台处理问题生成的函数"""
    logger.info(f"开始处理问题生成任务 {task_id}")
    
    # 初始化任务状态
    current_tasks[task_id]['status'] = 'processing'
    current_tasks[task_id]['is_streaming'] = True
    current_tasks[task_id]['last_update'] = time.time()
    
    # 创建API客户端
    if model_provider == "volcano":
        client = Ark(
            base_url=base_url,
            api_key=api_key
        )
        actual_model = model_name
    else:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        actual_model = "gpt-3.5-turbo"
    
    try:
        logger.info(f"发送生成问题请求，温度: {temperature}")
        
        # 发送流式请求
        if model_provider == "volcano":
            stream = client.chat.completions.create(
                model=actual_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(temperature),
                stream=True  # 启用流式输出
            )
        else:
            stream = client.chat.completions.create(
                model=actual_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(temperature),
                stream=True  # 启用流式输出
            )
        
        # 实时收集流式响应
        generated_text = ""
        for chunk in stream:
            # 检查是否暂停
            if current_tasks[task_id].get('paused', False):
                logger.info(f"任务 {task_id} 已暂停，等待继续...")
                while current_tasks[task_id].get('paused', False):
                    time.sleep(1)
                    if task_id not in current_tasks:
                        logger.info(f"任务 {task_id} 已被删除，终止处理")
                        return []
                logger.info(f"任务 {task_id} 已恢复")
            
            # 提取内容
            if model_provider == "volcano":
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    generated_text += chunk.choices[0].delta.content
            else:
                if chunk.choices and chunk.choices[0].delta and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    generated_text += chunk.choices[0].delta.content
            
            # 实时解析已有文本中的问题
            current_questions = []
            lines = generated_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                # 匹配格式为"1. 问题"或"1.问题"或"1、问题"的行
                if re.match(r'^\d+[\.\、] .+', line) or re.match(r'^\d+\..+', line):
                    # 去除序号和点，保留问题内容
                    question = re.sub(r'^\d+[\.\、]\s*', '', line).strip()
                    if question:
                        current_questions.append(question)
            
            # 更新任务状态
            if current_questions:
                # 限制问题数量
                current_questions = current_questions[:current_tasks[task_id]['num_questions']]
                current_tasks[task_id]['results'] = current_questions
                current_tasks[task_id]['completed'] = len(current_questions)
                current_tasks[task_id]['last_update'] = time.time()
        
        # 流式生成完成后，做最后的处理
        logger.info(f"问题生成流式响应完成，处理最终结果文本")
        
        # 最终处理生成的文本
        final_questions = []
        lines = generated_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # 匹配格式为"1. 问题"或"1.问题"或"1、问题"的行
            if re.match(r'^\d+[\.\、] .+', line) or re.match(r'^\d+\..+', line):
                # 去除序号和点，保留问题内容
                question = re.sub(r'^\d+[\.\、]\s*', '', line).strip()
                if question:
                    final_questions.append(question)
        
        # 如果没有匹配到正确格式的问题，尝试直接按行分割
        if not final_questions:
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('---'):
                    final_questions.append(line)
        
        # 更新最终结果
        results = final_questions[:current_tasks[task_id]['num_questions']]
        
        current_tasks[task_id]['completed'] = len(results)
        current_tasks[task_id]['results'] = results
        current_tasks[task_id]['status'] = 'completed'
        current_tasks[task_id]['is_streaming'] = False  # 流式处理结束
        current_tasks[task_id]['last_update'] = time.time()
        
        # 保存结果
        save_generation_results(task_id, results)
        
        logger.info(f"生成问题任务 {task_id} 完成，共生成 {len(results)} 个问题")
        
    except Exception as e:
        error_msg = f"生成问题时出错: {str(e)}"
        logger.error(error_msg)
        current_tasks[task_id]['status'] = 'failed'
        current_tasks[task_id]['error'] = error_msg
        current_tasks[task_id]['is_streaming'] = False
        current_tasks[task_id]['last_update'] = time.time()
    
    return results

def save_generation_results(task_id, results):
    """保存问题生成结果到文件"""
    filename = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存任务元数据
    if task_id in current_tasks:
        metadata_file = os.path.join(app.config['RESULTS_FOLDER'], f'{task_id}_metadata.json')
        metadata = {
            'generation_prompt': current_tasks[task_id].get('generation_prompt', ''),
            'gen_system_prompt': current_tasks[task_id].get('gen_system_prompt', ''),
            'gen_temperature': current_tasks[task_id].get('gen_temperature', 0.8),
            'model_provider': current_tasks[task_id].get('model_provider', 'volcano'),
            'model_name': current_tasks[task_id].get('model_name', ''),
            'timestamp': time.time()
        }
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存生成任务元数据出错: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)