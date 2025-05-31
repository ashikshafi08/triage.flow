from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Setup WebDriver
driver = webdriver.Chrome()
driver.implicitly_wait(10)

try:
    # Open the application
    driver.get("http://localhost:3000")
    
    # Navigate to issue creation page
    issue_form = driver.find_element(By.LINK_TEXT, "Create Session")
    issue_form.click()
    
    # Fill issue details
    issue_url = driver.find_element(By.NAME, "issue_url")
    issue_url.send_keys("https://github.com/owner/repo/issues/123")
    
    # Select prompt type
    prompt_type = driver.find_element(By.NAME, "prompt_type")
    for option in prompt_type.find_elements(By.TAG_NAME, "option"):
        if option.text == "Explain":
            option.click()
            break
    
    # Submit form
    submit_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    submit_btn.click()
    
    # Wait for chat interface
    chat_window = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "chat-container"))
    )
    
    # Send test message
    input_field = driver.find_element(By.CSS_SELECTOR, "input[placeholder='Type your message...']")
    input_field.send_keys("Explain the root cause")
    input_field.send_keys(Keys.RETURN)
    
    # Verify response
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, "//div[contains(text(),'Assistant:')]"))
    )
    
    print("Chat interface test passed!")
    
except Exception as e:
    print(f"Test failed: {str(e)}")
    raise
finally:
    driver.quit()
