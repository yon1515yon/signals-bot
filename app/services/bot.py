import logging
import time

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from app.config import settings

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


REPLY_KEYBOARD = [
    ["🔥 Топ-5 Сигналов", "📊 Статистика"],  
    ["📈 Список акций", "🏛 Уровни по тикеру"],  
]
MARKUP = ReplyKeyboardMarkup(REPLY_KEYBOARD, one_time_keyboard=False, resize_keyboard=True)
API_BASE_URL = settings.API_BASE_URL


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    async with httpx.AsyncClient() as client:
        await client.post(f"{API_BASE_URL}/portfolio/create/{user_id}")

    await update.message.reply_text("Привет! Твой виртуальный портфель на 1 000 000 руб. создан.", reply_markup=MARKUP)


async def get_top_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Запрашивает сигналы у нашего API и отправляет их пользователю по одному."""
    await update.message.reply_text("🔍 Ищу самые горячие сигналы, подождите...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{API_BASE_URL}/signals")
            response.raise_for_status()
            signals = response.json()

            if not signals:
                await update.message.reply_text(
                    "В данный момент нет активных сигналов. Сканер еще работает или рынок спокоен."
                )
                return

            await update.message.reply_text("🔥 **Топ-5 Горячих Сигналов:**", parse_mode="Markdown")

            for signal in signals[:5]:
                description = signal.get("description", "Нет описания.")
                risk = signal["details"].get("risk_score", "N/A")
                win_rate = signal["details"].get("model_win_rate", "N/A") 
                signal_id = signal["id"]

                message = f"Ticker: *{signal['ticker']}*\n"
                message += f"Тип: _{signal['signal_type']}_\n"
                message += f"Риск: *{risk}/10*\n"
                message += f"AI WinRate: *{win_rate}%* (WFA)\n" 
                message += f"Описание: {description}\n\n"

                keyboard = [
                    [
                        InlineKeyboardButton("✅ Успех", callback_data=f"feedback_{signal_id}_SUCCESS"),
                        InlineKeyboardButton("❌ Провал", callback_data=f"feedback_{signal_id}_FAIL"),
                        InlineKeyboardButton("➖ Б/У", callback_data=f"feedback_{signal_id}_BREAKEVEN"),
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await update.message.reply_html(message, reply_markup=reply_markup)

        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка API при запросе сигналов: {e}")
            await update.message.reply_text("Не удалось получить сигналы от сервера. Возможно, он еще не готов.")
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при получении сигналов: {e}")
            await update.message.reply_text("Произошла внутренняя ошибка.")


async def get_key_levels(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Запрашивает ключевые уровни для тикера."""
    try:
        ticker = context.args[0].upper()
    except (IndexError, ValueError):
        await update.message.reply_text("Пожалуйста, укажите тикер после команды. Пример: /levels SBER")
        return

    await update.message.reply_text(f"🔍 Ищу уровни для *{ticker}*...", parse_mode="Markdown")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{API_BASE_URL}/levels/{ticker}")
            response.raise_for_status()
            levels = response.json()

            message = f"**Ключевые зоны для {ticker}:**\n\n"
            resistances = [lvl for lvl in levels if lvl["level_type"] == "resistance"]
            supports = [lvl for lvl in levels if lvl["level_type"] == "support"]

            if resistances:
                message += "🔴 *Сопротивление:*\n"
                for level in resistances:
                    message += f"  - Зона: `{level['start_price']:.2f} - {level['end_price']:.2f}` (Интенс.: {level['intensity']}%)\n"

            if supports:
                message += "\n🟢 *Поддержка:*\n"
                for level in sorted(supports, key=lambda x: x["start_price"], reverse=True):
                    message += f"  - Зона: `{level['start_price']:.2f} - {level['end_price']:.2f}` (Интенс.: {level['intensity']}%)\n"

            await update.message.reply_html(message)

        except httpx.HTTPStatusError:
            await update.message.reply_text(
                f"Не удалось найти ключевые уровни для тикера *{ticker}*. Убедитесь, что тикер верный и данные для него рассчитаны.",
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error(f"Ошибка при получении уровней для {ticker}: {e}")
            await update.message.reply_text("Произошла внутренняя ошибка.")

async def get_tickers_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Получает и отправляет список отслеживаемых акций."""
    await update.message.reply_text("🔍 Запрашиваю список отслеживаемых акций...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{API_BASE_URL}/tickers")
            response.raise_for_status()
            tickers = response.json()

            if not tickers:
                await update.message.reply_text(
                    "Список отслеживаемых акций пока пуст. Идет первоначальное сканирование."
                )
                return

            message_parts = ["*Отслеживаемые акции:*\n"]
            tickers_list = [f"`{t['ticker']}`" for t in tickers]

            for i in range(0, len(tickers_list), 5):
                message_parts.append(" ".join(tickers_list[i : i + 5]))

            await update.message.reply_text("\n".join(message_parts), parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Ошибка при получении списка тикеров: {e}")
            await update.message.reply_text("Не удалось получить список акций от сервера.")


async def get_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Запрашивает статистику по сигналам у API и отправляет пользователю."""

    await update.message.reply_text("📊 Запрашиваю статистику, минутку...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{API_BASE_URL}/stats")
            response.raise_for_status()
            stats = response.json()

            if stats["total_signals"] == 0:
                await update.message.reply_text(
                    "Пока нет ни одного оцененного сигнала. Используйте кнопки под сигналами, чтобы оставить свой отзыв!"
                )
                return

            message = (
                f"📊 **Статистика Эффективности Сигналов:**\n\n"
                f"Всего оценено сигналов: *{stats['total_signals']}*\n"
                f"✅ Успешных: *{stats['success_count']}*\n"
                f"❌ Неуспешных: *{stats['fail_count']}*\n"
                f"➖ В безубытке: *{stats['breakeven_count']}*\n\n"
                f"🎯 **Процент успеха (Winrate): {stats['success_rate']}%**"
            )

            await update.message.reply_html(message)

        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка API при запросе статистики: {e}")
            await update.message.reply_text("Не удалось получить статистику от сервера. Возможно, он еще не готов.")
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при получении статистики: {e}")
            await update.message.reply_text("Произошла внутренняя ошибка при запросе статистики.")


async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатия на inline-кнопки обратной связи."""
    query = update.callback_query
    await query.answer() 

    try:
        _, signal_id_str, reaction = query.data.split("_")
        signal_id = int(signal_id_str)
        user_id = query.from_user.id
    except (ValueError, IndexError):
        logger.error(f"Некорректные callback_data: {query.data}")
        await query.edit_message_text(text=f"{query.message.text}\n\nОшибка обработки отзыва.")
        return

    feedback_data = {"signal_id": signal_id, "user_id": user_id, "reaction": reaction}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_BASE_URL}/feedback", json=feedback_data)
            response.raise_for_status()

            feedback_status_text = ""
            if reaction == "SUCCESS":
                feedback_status_text = "✅ Отмечено как Успех. Спасибо!"
            elif reaction == "FAIL":
                feedback_status_text = "❌ Отмечено как Провал. Спасибо!"
            elif reaction == "BREAKEVEN":
                feedback_status_text = "➖ Отмечено как Б/У. Спасибо!"

            await query.edit_message_text(
                text=f"{query.message.text}\n\n*{feedback_status_text}*", parse_mode="Markdown"
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка API при отправке отзыва: {e.response.text}")
            await query.edit_message_text(
                text=f"{query.message.text}\n\nНе удалось сохранить ваш отзыв. Попробуйте позже."
            )
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при отправке отзыва: {e}")
            await query.edit_message_text(text=f"{query.message.text}\n\nПроизошла внутренняя ошибка.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает нажатия на кнопки меню."""
    text = update.message.text
    if text == "🔥 Топ-5 Сигналов":
        await get_top_signals(update, context)
    elif text == "📈 Список акций":
        await get_tickers_list(update, context)
    elif text == "🏛 Уровни по тикеру":  
        await update.message.reply_text(
            "Чтобы получить уровни, введите команду в формате: /levels <ТИКЕР>\nНапример: /levels SBER"
        )
    elif text == "📊 Статистика":  
        await get_stats(update, context)
    else:
        await update.message.reply_text("Неизвестная команда. Используйте кнопки меню или введите /start для помощи.")


async def get_forecast_details(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет детальный прогноз с графиком для указанного тикера."""
    try:
        ticker = context.args[0].upper()
    except (IndexError, ValueError):
        await update.message.reply_text("Пожалуйста, укажите тикер после команды. Пример: /forecast SBER")
        return

    await update.message.reply_text(f"📈 Строю прогноз для *{ticker}*, подождите...", parse_mode="Markdown")

    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            plot_response = await client.get(f"{API_BASE_URL}/plots/forecast/{ticker}")

            plot_response.raise_for_status()
            image_bytes = await plot_response.aread()


            forecast_data_response = await client.get(f"{API_BASE_URL}/forecast/{ticker}")
            forecast_data_response.raise_for_status()
            forecasts = forecast_data_response.json()

            current_price = forecasts[0]["forecast_value"]  
            price_in_7_days = forecasts[6]["forecast_value"]
            price_in_30_days = forecasts[-1]["forecast_value"]

            change_7d_pct = (price_in_7_days / current_price - 1) * 100
            change_30d_pct = (price_in_30_days / current_price - 1) * 100

            caption = (
                f"**Прогноз для {ticker}**\n\n"
                f"Текущая цена (прогноз на сегодня): *{current_price:.2f} руб.*\n\n"
                f"Прогноз на 7 дней: *{price_in_7_days:.2f} руб.* ({change_7d_pct:+.2f}%)\n"
                f"Прогноз на 30 дней: *{price_in_30_days:.2f} руб.* ({change_30d_pct:+.2f}%)\n"
            )

            await update.message.reply_photo(photo=image_bytes, caption=caption, parse_mode="Markdown")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                await update.message.reply_text(
                    f"Не удалось найти прогноз для *{ticker}*. Убедитесь, что тикер верный.", parse_mode="Markdown"
                )
            else:
                logger.error(f"Ошибка API при запросе прогноза для {ticker}: {e.response.text}")
                await update.message.reply_text("Не удалось получить прогноз от сервера.")
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при получении прогноза для {ticker}: {e}")
            await update.message.reply_text("Произошла внутренняя ошибка.")


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /subscribe."""
    user = update.effective_user
    user_data = {"user_id": user.id, "username": user.username}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_BASE_URL}/subscribe", json=user_data)
            response.raise_for_status()
            api_response = response.json()
            await update.message.reply_text(api_response.get("message", "Готово!"))
        except Exception as e:
            logger.error(f"Ошибка при подписке пользователя {user.id}: {e}")
            await update.message.reply_text("Не удалось обработать подписку. Попробуйте позже.")


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /unsubscribe."""
    user = update.effective_user
    user_data = {"user_id": user.id}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_BASE_URL}/unsubscribe", json=user_data)
            response.raise_for_status()
            api_response = response.json()
            await update.message.reply_text(api_response.get("message", "Готово!"))
        except Exception as e:
            logger.error(f"Ошибка при отписке пользователя {user.id}: {e}")
            await update.message.reply_text("Не удалось обработать отписку. Попробуйте позже.")


async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE_URL}/portfolio/{user_id}")
        if resp.status_code != 200:
            await update.message.reply_text("Ошибка получения портфеля.")
            return

        data = resp.json()
        if not data:
            await update.message.reply_text("Портфель пуст.")
            return

        # Красивый вывод
        msg = "💼 <b>МОЙ ПОРТФЕЛЬ</b>\n\n"
        msg += f"💰 <b>Баланс:</b> {data['equity']:,.2f} ₽\n"
        msg += f"💵 <b>Кэш:</b> {data['cash']:,.2f} ₽\n"

        color = "🟢" if data["pnl"] >= 0 else "🔴"
        msg += f"{color} <b>PnL:</b> {data['pnl']:,.2f} ₽ ({data['pnl_pct']}%) \n\n"

        msg += "<b>Активные позиции:</b>\n"
        for p in data["positions"]:
            p_color = "🟢" if p["pnl"] >= 0 else "🔴"
            msg += f"{p_color} <b>{p['ticker']}</b>: {p['pnl_pct']}%\n"
            msg += f"   Вход: {p['entry']} -> {p['current']}\n"
            msg += f"   /close_{p['id']} (Закрыть)\n\n"

    await update.message.reply_html(msg)


async def handle_trading_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data

    if data.startswith("buy_"):

        _, ticker, price = data.split("_")
        user_id = query.from_user.id

        payload = {"ticker": ticker, "price": float(price), "amount": 50000}  # Фикс сумма

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{API_BASE_URL}/portfolio/trade/{user_id}", json=payload)
            if resp.status_code == 200:
                await query.answer("Ордер исполнен!", show_alert=True)
            else:
                await query.answer(f"Ошибка: {resp.text}", show_alert=True)


def main() -> None:
    """Запуск бота."""
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.error("Токен TELEGRAM_BOT_TOKEN не найден! Бот не может быть запущен.")
        return

    application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("signals", get_top_signals))
    application.add_handler(CommandHandler("levels", get_key_levels))
    application.add_handler(CommandHandler("tickers", get_tickers_list))
    application.add_handler(CommandHandler("stats", get_stats))
    application.add_handler(CommandHandler("forecast", get_forecast_details))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe))

    application.add_handler(CallbackQueryHandler(handle_feedback, pattern=r"^feedback_"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Бот запущен и готов к работе...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)




if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Бот упал с критической ошибкой: {e}")
        raise e
