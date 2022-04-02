import React from 'react';
import './App.css';
import { Autocomplete, Button, Checkbox, FormControl, FormControlLabel, FormLabel, Radio, RadioGroup, TextField } from '@mui/material';
import User from './models/User';
import CEvent from './models/CustomEvent';
import EventCard from './components/EventCard/EventCard';

const API_URL = 'http://127.0.0.1:5000';

function App() {
  const [modelVariant, setmodelVariant] = React.useState<'basic' | 'svd'>('basic');
  const [trainChecked, setTrainChecked] = React.useState<boolean>(false);
  const [userId, setUserId] = React.useState<string>('');
  // user ids for the search dropdown
  const [users, setUsers] = React.useState<string[]>([]);
  const [user, setUser] = React.useState<User | null>(null);
  const [events, setEvents] = React.useState<CEvent[]>([]);

  const fetchUsers = async () => {
    const response = await fetch(`${API_URL}/users?id=${userId}&train=${trainChecked}`);
    const data = await response.json();
    setUsers(data);
  }

  const fetchUser = async () => {
    const response = await fetch(`${API_URL}/user?id=${userId}`);
    const data: User[] = await response.json();
    setUser(data[0]); // return type is an array with always only 1 user
  }

  const fetchEvents = async () => {
    const response = await fetch(`${API_URL}/events?id=${userId}&svd=${modelVariant === 'svd'}`);
    const data: CEvent[] = await response.json();
    data.sort((a, b) => a.rating! - b.rating!).reverse();
    setEvents(data);
  }

  const handleUserIdChange = (event: any, newUserId: string) => {
    setUserId(newUserId);
  }

  const handleModelVariantChange = (variant: 'basic' | 'svd') => {
    setmodelVariant(variant);
  }

  const handleTrainFilterChecked = (event: any) => {
    setTrainChecked(event.target.checked);
  }

  const runModel = () => {
    fetchUser();
    fetchEvents();
  }

  React.useEffect(() => {
    fetchUsers();
  }, [userId]);

  return (
    <div className="App">
      <div className='left'>
        <h1>Recommender</h1>
        <FormControl>
          <RadioGroup row defaultValue="basic" name="model">
            <FormControlLabel value="basic" onChange={_ => handleModelVariantChange('basic')} control={<Radio />} label="Basic" />
            <FormControlLabel value="svd" onChange={_ => handleModelVariantChange('svd')} control={<Radio />} label="SVD" />
          </RadioGroup>
          <Autocomplete id="input-field" options={users}
            inputValue={userId}
            getOptionLabel={(option: string) => option.toString()}
            onInputChange={handleUserIdChange}
            renderInput={params =>
              <TextField {...params} label="User ID" />}
          />
          <FormControlLabel onChange={handleTrainFilterChecked} control={<Checkbox />} label="Show users in training set only" />
          <Button variant="contained" disabled={userId.length <= 0} onClick={runModel}>Search</Button>
        </FormControl>
        {user && <div>
          <h2>User {user.user_id}</h2>
          <p>{user.birthyear}</p>
          <p>{user.gender}</p>
          <p>{user.joinedAt}</p>
          <p>{user.location}</p>
        </div>}
      </div>
      {events.length > 0 &&
        <div className='right'>
          {events.map(event =>
            <EventCard event={event} />
          )}
        </div>
      }
    </div>
  );
}

export default App;
