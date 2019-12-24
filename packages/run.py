from api.app import create_app

application = create_app(
    config_name='prod')

if __name__ == '__main__':
    application.run()
